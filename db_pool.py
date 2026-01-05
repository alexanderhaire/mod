"""
Database Connection Pool

Provides thread-safe connection pooling for SQL Server connections.
Reduces overhead of creating new connections for each request.
"""

import logging
import threading
import time
from contextlib import contextmanager
from queue import Queue, Empty
from typing import Optional

import pyodbc

from secrets_loader import build_connection_string

LOGGER = logging.getLogger(__name__)


class ConnectionPool:
    """
    Thread-safe connection pool for pyodbc connections.
    
    Usage:
        pool = ConnectionPool(min_size=2, max_size=10)
        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
    """
    
    def __init__(
        self,
        min_size: int = 2,
        max_size: int = 10,
        connection_timeout: int = 30,
        idle_timeout: int = 300,
    ):
        """
        Initialize the connection pool.
        
        Args:
            min_size: Minimum connections to maintain
            max_size: Maximum connections allowed
            connection_timeout: Seconds to wait for a connection
            idle_timeout: Seconds before idle connections are closed
        """
        self.min_size = min_size
        self.max_size = max_size
        self.connection_timeout = connection_timeout
        self.idle_timeout = idle_timeout
        
        self._pool: Queue = Queue(maxsize=max_size)
        self._size = 0
        self._lock = threading.Lock()
        self._connection_string: Optional[str] = None
        self._initialized = False
        
    def _get_connection_string(self) -> str:
        """Get or cache the connection string."""
        if self._connection_string is None:
            conn_str, server, database, auth = build_connection_string()
            self._connection_string = conn_str
            LOGGER.info(f"Connection pool configured for {server}/{database} ({auth})")
        return self._connection_string
    
    def _create_connection(self) -> pyodbc.Connection:
        """Create a new database connection."""
        conn_str = self._get_connection_string()
        try:
            conn = pyodbc.connect(conn_str, timeout=self.connection_timeout)
            conn.autocommit = True  # Default to autocommit for read operations
            LOGGER.debug("Created new database connection")
            return conn
        except pyodbc.Error as e:
            LOGGER.error(f"Failed to create connection: {e}")
            raise
    
    def _validate_connection(self, conn: pyodbc.Connection) -> bool:
        """Check if a connection is still valid."""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception:
            return False
    
    def initialize(self) -> None:
        """Pre-populate the pool with minimum connections."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            LOGGER.info(f"Initializing connection pool (min={self.min_size}, max={self.max_size})")
            for _ in range(self.min_size):
                try:
                    conn = self._create_connection()
                    self._pool.put((conn, time.time()))
                    self._size += 1
                except Exception as e:
                    LOGGER.warning(f"Failed to pre-create connection: {e}")
                    break
            
            self._initialized = True
            LOGGER.info(f"Connection pool initialized with {self._size} connections")
    
    def acquire(self) -> pyodbc.Connection:
        """
        Acquire a connection from the pool.
        
        Returns a connection, creating a new one if necessary.
        """
        self.initialize()
        
        start_time = time.time()
        
        while True:
            # Try to get from pool
            try:
                conn, created_at = self._pool.get_nowait()
                
                # Check if connection is still valid
                if self._validate_connection(conn):
                    LOGGER.debug("Acquired connection from pool")
                    return conn
                else:
                    # Connection is stale, close it
                    try:
                        conn.close()
                    except Exception:
                        pass
                    with self._lock:
                        self._size -= 1
                    continue
                    
            except Empty:
                # Pool is empty, try to create new connection
                with self._lock:
                    if self._size < self.max_size:
                        self._size += 1
                        try:
                            conn = self._create_connection()
                            LOGGER.debug("Created new connection (pool was empty)")
                            return conn
                        except Exception:
                            self._size -= 1
                            raise
                
                # Pool is at max capacity, wait for a connection
                elapsed = time.time() - start_time
                if elapsed >= self.connection_timeout:
                    raise TimeoutError(
                        f"Could not acquire connection within {self.connection_timeout}s"
                    )
                
                try:
                    conn, created_at = self._pool.get(timeout=1)
                    if self._validate_connection(conn):
                        return conn
                except Empty:
                    continue
    
    def release(self, conn: pyodbc.Connection) -> None:
        """Return a connection to the pool."""
        try:
            # Reset connection state
            conn.rollback()
        except Exception:
            # Connection is broken, close it
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._size -= 1
            LOGGER.debug("Closed broken connection")
            return
        
        # Return to pool
        try:
            self._pool.put_nowait((conn, time.time()))
            LOGGER.debug("Released connection to pool")
        except Exception:
            # Pool is full somehow, close connection
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._size -= 1
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for acquiring and releasing connections.
        
        Usage:
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                ...
        """
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        LOGGER.info("Closing all connections in pool")
        with self._lock:
            while not self._pool.empty():
                try:
                    conn, _ = self._pool.get_nowait()
                    conn.close()
                except Exception:
                    pass
            self._size = 0
            self._initialized = False
    
    @property
    def available_connections(self) -> int:
        """Number of connections available in the pool."""
        return self._pool.qsize()
    
    @property
    def total_connections(self) -> int:
        """Total number of connections (in use + available)."""
        return self._size


# Global connection pool instance
_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_pool(
    min_size: int = 2,
    max_size: int = 10,
) -> ConnectionPool:
    """Get or create the global connection pool."""
    global _pool
    
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(min_size=min_size, max_size=max_size)
    
    return _pool


def get_connection():
    """
    Convenience function to get a connection context manager.
    
    Usage:
        with get_connection() as conn:
            cursor = conn.cursor()
            ...
    """
    return get_pool().get_connection()


def get_cursor():
    """
    Convenience function to get a cursor context manager.
    
    Usage:
        with get_cursor() as cursor:
            cursor.execute("SELECT 1")
    """
    @contextmanager
    def cursor_context():
        with get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
            finally:
                cursor.close()
    
    return cursor_context()
