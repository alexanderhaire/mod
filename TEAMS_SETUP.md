# Teams Notification Setup Guide

## Overview

The reorder point notification system sends real-time alerts to Microsoft Teams when inventory items hit their reorder point. This guide will walk you through setting up the Teams webhook and testing the system.

## Step 1: Create Teams Incoming Webhook

1. **Open Microsoft Teams** and navigate to the channel where you want to receive notifications

2. **Click the "..." (More options)** next to the channel name

3. **Select "Connectors"** (or "Workflows" in newer Teams versions)

4. **Search for "Incoming Webhook"** and click "Add" or "Configure"

5. **Give your webhook a name**, such as:
   - "Reorder Point Alerts"
   - "Inventory Notifications"

6. **Optionally upload an icon** for the webhook (makes messages more recognizable)

7. **Click "Create"** and copy the webhook URL
   - It will look like: `https://your-tenant.webhook.office.com/webhookb2/...`
   - **IMPORTANT:** Keep this URL secure - anyone with it can post to your channel

## Step 2: Configure secrets.toml

1. **Open the file:** `.streamlit/secrets.toml` in your project directory

2. **Add the Teams configuration section** (or update if it exists):

```toml
[teams]
webhook_url = "PASTE_YOUR_WEBHOOK_URL_HERE"
enabled = true
check_interval_minutes = 30
```

3. **Replace** `PASTE_YOUR_WEBHOOK_URL_HERE` with the actual webhook URL from Step 1

4. **Save the file**

### Configuration Options

- `webhook_url` - The Teams incoming webhook URL (required)
- `enabled` - Set to `false` to disable notifications without removing the webhook
- `check_interval_minutes` - How often to check for reorder point changes (default: 30)

## Step 3: Test the Webhook

Run the test command to verify your Teams integration is working:

```bash
python scripts/start_reorder_monitor.py --test
```

**Expected output:**
```
===============================================================
TEAMS WEBHOOK TEST MODE
===============================================================
Sending test message to Teams...
✅ Test message sent successfully!
Check your Teams channel to verify receipt
```

**Check your Teams channel** - you should see a green test message appear.

### Troubleshooting Test Failures

If the test fails:

1. **Verify the webhook URL** is correct in `secrets.toml`
   - No extra spaces or quotes
   - Includes the full URL starting with `https://`

2. **Check network connectivity**
   - Ensure your computer can reach `outlook.office365.com`
   - Check firewall/proxy settings if behind corporate network

3. **Verify the webhook is still active**
   - Webhooks can be revoked in Teams settings
   - Recreate the webhook if needed

## Step 4: Run a Manual Check

Test the actual reorder point monitoring (won't send notifications unless items are below reorder point):

```bash
python scripts/start_reorder_monitor.py --once
```

**Expected output:**
```
===============================================================
Starting reorder point check at 2026-02-12 14:30:00
===============================================================
Querying current reorder recommendations from database...
Retrieved 145 items from database
Current status: 3 Critical, 5 Soon, 137 OK
Loading previous state...
No previous state file found, starting fresh
Detecting state transitions...
Found 8 items requiring notification
Sending notification for NPKACEK (Critical, 5.2 days coverage)
Successfully sent Teams alert for NPKACEK
...
Check cycle completed successfully
```

**Check your Teams channel** - you should see notification cards for any items below reorder point.

## Step 5: Start Continuous Monitoring

### Option A: Manual Start (Testing)

Run the monitor in a terminal window:

```bash
python scripts/start_reorder_monitor.py
```

This will run continuously, checking every 30 minutes. Press Ctrl+C to stop.

### Option B: Windows Task Scheduler (Production)

For automatic startup and background operation:

1. **Open Task Scheduler** (Win + R, type `taskschd.msc`)

2. **Create a new task:**
   - Name: "Reorder Point Monitor"
   - Description: "Monitors inventory reorder points and sends Teams notifications"

3. **Triggers tab:**
   - New trigger → "At startup"
   - Or use "At log on" if you want it to run when you log in

4. **Actions tab:**
   - Action: "Start a program"
   - Program: `C:\Users\alexh\Downloads\mod\scripts\run_monitor.bat`
   - Start in: `C:\Users\alexh\Downloads\mod`

5. **Conditions tab:**
   - Uncheck "Start only if on AC power" (if on a laptop)

6. **Settings tab:**
   - Check "Restart on failure"
   - Restart every: 5 minutes
   - Attempt up to: 3 times

7. **Click OK** and test by running the task manually

### Option C: Batch File (Quick Start)

Double-click `scripts\run_monitor.bat` to start the monitor in the background.

Log output will be saved to `logs\reorder_monitor.log`

## Notification Behavior

### What Triggers a Notification?

Notifications are sent ONLY when:

- ✅ An item **newly** drops below its reorder point (OK → Soon or Critical)
- ✅ An item **escalates** in urgency (Soon → Critical)

Notifications are NOT sent when:

- ❌ An item remains at the same urgency level (already notified)
- ❌ An item is **improving** (Critical → Soon, or Soon → OK)
- ❌ An item is already below reorder point from the previous check

### Message Content

Each notification includes:

- Item number and description
- On-hand quantity
- On-order quantity
- Available quantity (on-hand + on-order)
- Days of coverage remaining
- Urgency level (Critical or Soon)
- Link to view full reorder dashboard

### Color Coding

- 🚨 **Red** - Critical (order today)
- ⚠️ **Orange** - Soon (order within 7 days)

## Monitoring and Logs

### Check Monitor Status

View the log file:

```bash
type logs\reorder_monitor.log
```

Or tail the log in real-time (PowerShell):

```powershell
Get-Content logs\reorder_monitor.log -Wait -Tail 20
```

### Log Locations

- **Monitor logs:** `logs/reorder_monitor.log`
- **State history:** `data/reorder_state_history.json` (tracks previous state)

### State History

The system maintains a state file (`data/reorder_state_history.json`) to track:

- Which items are currently below reorder point
- Their urgency levels
- When state was last updated

**Don't edit this file manually** - it's automatically managed by the system.

## Advanced Configuration

### Change Check Interval

Edit `.streamlit/secrets.toml`:

```toml
[teams]
check_interval_minutes = 15  # Check every 15 minutes instead of 30
```

Or pass it as a command-line argument:

```bash
python scripts/start_reorder_monitor.py --interval 15
```

### Disable Notifications Temporarily

Edit `.streamlit/secrets.toml`:

```toml
[teams]
enabled = false  # Temporarily disable without removing webhook
```

### Multiple Channels

To send notifications to multiple Teams channels:

1. Create multiple webhooks in different channels
2. You'll need to modify the code to support multiple URLs
3. Or use Teams channel forwarding rules

## Troubleshooting

### No Notifications Being Sent

1. **Check if items are actually below reorder point:**
   ```bash
   python scripts/start_reorder_monitor.py --once
   ```
   Look for "Found X items requiring notification"

2. **Check if notifications are enabled:**
   ```bash
   type .streamlit\secrets.toml | findstr enabled
   ```
   Should show `enabled = true`

3. **Check the state file:**
   ```bash
   type data\reorder_state_history.json
   ```
   If items are already in "Critical" or "Soon" state, they won't notify again

4. **Reset state to test:**
   ```bash
   del data\reorder_state_history.json
   python scripts/start_reorder_monitor.py --once
   ```
   This will treat all items as new and send notifications

### Monitor Not Running

1. **Check Task Scheduler** (if using scheduled task)
   - Is the task enabled?
   - Is it showing as "Running"?
   - Check "Last Run Result" (should be 0 for success)

2. **Check logs:**
   ```bash
   type logs\reorder_monitor.log
   ```

3. **Test manually:**
   ```bash
   python scripts/start_reorder_monitor.py --once
   ```

### Webhook URL Changed/Revoked

If the webhook is deleted or changed in Teams:

1. **Create a new webhook** in Teams (Step 1)
2. **Update secrets.toml** with the new URL (Step 2)
3. **Test the new webhook:**
   ```bash
   python scripts/start_reorder_monitor.py --test
   ```

## Security Notes

- **Never commit the webhook URL to git** - it's already in `.gitignore`
- **Rotate the webhook URL** if it's exposed or compromised
- **Limit access** to `.streamlit/secrets.toml` on shared systems
- **Use dedicated channels** - don't use your main team channel if you want to limit who sees alerts

## Support

For issues or questions:

1. Check the logs: `logs/reorder_monitor.log`
2. Run with `--test` flag to verify webhook
3. Run with `--once` flag to test a single check cycle
4. Check that database connection is working
5. Verify Teams webhook is still active in Teams connector settings

## Quick Command Reference

```bash
# Test webhook
python scripts/start_reorder_monitor.py --test

# Run once (testing)
python scripts/start_reorder_monitor.py --once

# Run continuously (default 30 min interval)
python scripts/start_reorder_monitor.py

# Run continuously (custom interval)
python scripts/start_reorder_monitor.py --interval 15

# View logs
type logs\reorder_monitor.log

# Reset state (force all items to notify again)
del data\reorder_state_history.json

# Start via batch file
scripts\run_monitor.bat
```
