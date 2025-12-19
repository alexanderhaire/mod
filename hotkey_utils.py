"""
Hotkey utility module for injecting Javascript keyboard shortcuts into Streamlit.
"""
import streamlit as st

def inject_hotkeys():
    """
    Injects Javascript to handle global hotkeys and Alt-Mode hints.
    
    Hotkeys:
    - Ctrl + / : Focus Chat Input
    - Ctrl + K : Trigger "New Chat"
    - Alt (Press): Toggle letter hints on/off.
    """
    
    js_script = """
    <script>
    (function() {
        // Safe access to parent document
        var doc = window.parent.document;
        var hintMap = {};
        var overlays = [];
        var isAltMode = false;

        console.log("Hotkeys: Initializing...");

        // -- STYLES --
        var styleId = 'hotkey-overlay-styles';
        if (!doc.getElementById(styleId)) {
            var style = doc.createElement('style');
            style.id = styleId;
            style.innerHTML = `
                .hotkey-badge {
                    position: absolute;
                    background-color: #ffb000; /* Terminal Amber */
                    color: #000000;
                    border: 1px solid #ffffff;
                    border-radius: 4px;
                    padding: 2px 5px;
                    font-family: monospace;
                    font-weight: bold;
                    font-size: 14px;
                    z-index: 2147483647; /* Max Z-Index */
                    box-shadow: 0 4px 8px rgba(0,0,0,0.8);
                    pointer-events: none;
                    text-transform: uppercase;
                }
            `;
            doc.head.appendChild(style);
        }

        // -- HELPER: GENERATE HINTS --
        function generateHints() {
            console.log("Hotkeys: Generating hints...");
            // clear old
            clearHints();

            // Find all visible buttons
            // selector includes buttons, submit inputs, role=button, radio/checkbox labels, text inputs, AND textareas
            var selector = 'button, input[type="submit"], input[type="text"], textarea, [role="button"], [role="radio"], [role="tab"], label[data-baseweb="radio"], label[data-baseweb="checkbox"]';
            var buttons = Array.from(doc.querySelectorAll(selector));
            
            // Also grab Streamlit specific radio/checkbox labels if they don't match above (often stMarkdownContainer inside a label)
            // But usually the label itself is the click target. 
            // Let's add standard labels that are children of stRadio or stCheckbox
            var extraLabels = Array.from(doc.querySelectorAll('div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label, div[data-testid="stTabs"] button'));
            buttons = buttons.concat(extraLabels);
            
            // Deduplicate in case of overlap
            buttons = buttons.filter(function(item, pos) {
                return buttons.indexOf(item) == pos;
            });
            
            // Filter visible
            var visibleButtons = buttons.filter(function(b) {
                var rect = b.getBoundingClientRect();
                var style = window.getComputedStyle(b);
                return rect.width > 0 && rect.height > 0 && 
                       style.display !== 'none' && 
                       style.visibility !== 'hidden' &&
                       style.opacity !== '0';
            });
            
            console.log("Hotkeys: Found " + visibleButtons.length + " visible buttons.");

            // alphabet: a-z, then za, zb... to allow many buttons
            var chars = 'abcdefghijklmnopqrstuvwxyz';
            
            visibleButtons.forEach(function(btn, index) {
                // Generate key
                var key = '';
                if (index < chars.length) {
                    key = chars[index];
                } else {
                    var first = Math.floor(index / chars.length) - 1;
                    var second = index % chars.length;
                    if (first < chars.length) {
                        key = chars[first] + chars[second];
                    } else {
                        key = '??'; 
                    }
                }

                // Create Badge
                var rect = btn.getBoundingClientRect();
                var badge = doc.createElement('div');
                badge.className = 'hotkey-badge';
                badge.innerText = key.toUpperCase();
                
                // Position absolute relative to document body
                // We must account for window scrolling
                var scrollTop = window.parent.scrollY || window.parent.pageYOffset || 0;
                var scrollLeft = window.parent.scrollX || window.parent.pageXOffset || 0;
                
                var top = rect.top + scrollTop;
                // Move to the right of the button: Left + Width + Padding
                var left = rect.left + rect.width + scrollLeft + 4; 
                
                // If off-screen to the right, flip to left? 
                // Simple check: window width
                if (left + 20 > window.parent.innerWidth) {
                     left = rect.left + scrollLeft - 25; // Left side
                }

                badge.style.top = top + 'px';
                badge.style.left = left + 'px';

                doc.body.appendChild(badge);
                
                overlays.push(badge);
                hintMap[key] = btn;
            });
            
            isAltMode = true;
        }

        function clearHints() {
            overlays.forEach(function(el) { el.remove(); });
            overlays = [];
            hintMap = {};
            isAltMode = false;
        }

        function toggleHints() {
            if (isAltMode) {
                clearHints();
            } else {
                generateHints();
            }
        }

        // -- HANDLERS --
        function handleKeyDown(e) {
            
            // ALT TOGGLE
            // We trigger on KeyUp to avoid repeat/hold issues? No, usually KeyDown is more responsive.
            // If user presses Alt, we toggle. 
            // Issue: Alt is a modifier.
            if (e.key === 'Alt') {
                e.preventDefault(); // Prevent browser menu focus if possible
                // We use a debounce or just toggle on down?
                // If we toggle on down, and they hold, repeat events fires.
                if (!e.repeat) {
                    toggleHints();
                }
                return;
            }
            
            // ESCAPE -> Clear
            if (e.key === 'Escape') {
                if (isAltMode) {
                    clearHints();
                    e.preventDefault(); 
                    return;
                }
            }

            // HOTKEY EXECUTION when in Alt Mode
            if (isAltMode) {
                // If they press a letter key, check map
                if (e.key.length === 1 && /[a-zA-Z]/.test(e.key) && !e.ctrlKey && !e.metaKey) {
                   var char = e.key.toLowerCase();
                   
                   // Check directly
                   if (hintMap[char]) {
                       console.log("Hotkeys: Clicking button for key " + char);
                       e.preventDefault();
                       e.stopPropagation();
                       hintMap[char].click();
                       
                       // Try to focus it too, if it's an input
                       try {
                           hintMap[char].focus();
                       } catch (err) {
                           // ignore
                       }
                       
                       clearHints(); // Dismiss after action
                       return;
                   } else {
                       // Key pressed but NOT in map.
                       // Should we consume it? Or let it type?
                       // User says: "I cant start typing"
                       // So: Dismiss mode, allow event to propagate.
                       console.log("Hotkeys: Key " + char + " not a hint. Dismissing.");
                       clearHints();
                       // Do NOT preventDefault. Let user type 'n' or whatever.
                       return;
                   }
                }
            }

            // GLOBAL HOTKEYS (CTRL combos)
            // CTRL + /
            if (e.ctrlKey && e.key === '/') {
                e.preventDefault();
                var chatInput = doc.querySelector('textarea[data-testid="stChatInput"]');
                if (chatInput) {
                    chatInput.focus();
                }
            }
            
            // CTRL + K
            if (e.ctrlKey && e.key === 'k') {
                e.preventDefault();
                var buttons = Array.from(doc.querySelectorAll('button'));
                var newChatBtn = buttons.find(function(btn) { 
                    return btn.innerText.includes("Restart Tour") || 
                           btn.innerText.includes("New Chat") || 
                           (btn.getAttribute("aria-label") || "").includes("New Chat");
                });
                
                if (newChatBtn) {
                    newChatBtn.click();
                }
            }
        }
        
        // Click anywhere to clear hints
        function handleClick(e) {
            if (isAltMode) {
                // If clicking a badge, let it pass? Badges are pointer-events: none
                // So click goes to element below.
                // We should clear hints on any click
                 // We rely on timeout? No, just clear.
                 // But if the click IS the action, we might race.
                 setTimeout(clearHints, 100);
            }
        }

        // Remove listeners
        doc.removeEventListener('keydown', window._hotkeyKeyHandler);
        doc.removeEventListener('click', window._hotkeyClickHandler);

        window._hotkeyKeyHandler = handleKeyDown;
        window._hotkeyClickHandler = handleClick;

        doc.addEventListener('keydown', handleKeyDown);
        doc.addEventListener('click', handleClick);

    })();
    </script>
    """
    
    st.components.v1.html(js_script, height=0, width=0)
