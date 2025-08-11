import ctypes
import ctypes.wintypes
import sys

# Constants for rotation
DMDO_DEFAULT = 0       # Landscape
DMDO_90 = 1            # Portrait
DMDO_180 = 2           # Landscape (flipped)
DMDO_270 = 3           # Portrait (flipped)

# Windows API constants
ENUM_CURRENT_SETTINGS = -1
CDS_UPDATEREGISTRY = 0x01
CDS_RESET = 0x40000000
DISP_CHANGE_SUCCESSFUL = 0
DISP_CHANGE_FAILED = -1
DISP_CHANGE_BADMODE = -2
DISP_CHANGE_RESTART = 1

def set_display_rotation(rotation: int):
    """
    Set display rotation using Windows API
    Args:
        rotation: 0=0°, 1=90°, 2=180°, 3=270°
    """
    try:
        # Check if we're on Windows
        if sys.platform != 'win32':
            print("❌ Screen rotation only works on Windows")
            return False
        
        user32 = ctypes.windll.user32
        
        # Define DEVMODE structure
        class DEVMODE(ctypes.Structure):
            _fields_ = [
                ('dmDeviceName', ctypes.c_wchar * 32),
                ('dmSpecVersion', ctypes.c_ushort),
                ('dmDriverVersion', ctypes.c_ushort),
                ('dmSize', ctypes.c_ushort),
                ('dmDriverExtra', ctypes.c_ushort),
                ('dmFields', ctypes.c_ulong),
                ('dmOrientation', ctypes.c_short),
                ('dmPaperSize', ctypes.c_short),
                ('dmPaperLength', ctypes.c_short),
                ('dmPaperWidth', ctypes.c_short),
                ('dmScale', ctypes.c_short),
                ('dmCopies', ctypes.c_short),
                ('dmDefaultSource', ctypes.c_short),
                ('dmPrintQuality', ctypes.c_short),
                ('dmColor', ctypes.c_short),
                ('dmDuplex', ctypes.c_short),
                ('dmYResolution', ctypes.c_short),
                ('dmTTOption', ctypes.c_short),
                ('dmCollate', ctypes.c_short),
                ('dmFormName', ctypes.c_wchar * 32),
                ('dmLogPixels', ctypes.c_ushort),
                ('dmBitsPerPel', ctypes.c_ulong),
                ('dmPelsWidth', ctypes.c_ulong),
                ('dmPelsHeight', ctypes.c_ulong),
                ('dmDisplayFlags', ctypes.c_ulong),
                ('dmDisplayFrequency', ctypes.c_ulong),
                ('dmICMMethod', ctypes.c_ulong),
                ('dmICMIntent', ctypes.c_ulong),
                ('dmMediaType', ctypes.c_ulong),
                ('dmDitherType', ctypes.c_ulong),
                ('dmReserved1', ctypes.c_ulong),
                ('dmReserved2', ctypes.c_ulong),
                ('dmPanningWidth', ctypes.c_ulong),
                ('dmPanningHeight', ctypes.c_ulong),
                ('dmDisplayOrientation', ctypes.c_ulong),
            ]

        # Get current display settings
        dm = DEVMODE()
        dm.dmSize = ctypes.sizeof(DEVMODE)
        dm.dmFields = 0x00000080  # DM_DISPLAYORIENTATION

        # Get current settings
        res = user32.EnumDisplaySettingsW(None, ENUM_CURRENT_SETTINGS, ctypes.byref(dm))
        if not res:
            print("❌ Failed to get current display settings")
            return False

        current_rotation = dm.dmDisplayOrientation
        print(f"📺 Current rotation: {current_rotation}")

        # Only change if different
        if current_rotation != rotation:
            print(f"🔄 Changing rotation from {current_rotation} to {rotation}")
            
            # Set new rotation
            dm.dmDisplayOrientation = rotation
            
            # Apply changes
            result = user32.ChangeDisplaySettingsW(ctypes.byref(dm), CDS_UPDATEREGISTRY | CDS_RESET)
            
            if result == DISP_CHANGE_SUCCESSFUL:
                print(f"✅ Screen rotated successfully to: {rotation}")
                return True
            elif result == DISP_CHANGE_RESTART:
                print("⚠️ Screen rotation applied, restart required")
                return True
            elif result == DISP_CHANGE_FAILED:
                print("❌ Failed to rotate screen - general failure")
                return False
            elif result == DISP_CHANGE_BADMODE:
                print("❌ Failed to rotate screen - bad mode")
                return False
            else:
                print(f"❌ Failed to rotate screen - error code: {result}")
                return False
        else:
            print(f"ℹ️ Screen already at rotation: {rotation}")
            return True
            
    except Exception as e:
        print(f"❌ Error during screen rotation: {e}")
        return False

def get_current_rotation():
    """Get current display rotation"""
    try:
        if sys.platform != 'win32':
            return None
        
        user32 = ctypes.windll.user32
        
        class DEVMODE(ctypes.Structure):
            _fields_ = [
                ('dmDeviceName', ctypes.c_wchar * 32),
                ('dmSpecVersion', ctypes.c_ushort),
                ('dmDriverVersion', ctypes.c_ushort),
                ('dmSize', ctypes.c_ushort),
                ('dmDriverExtra', ctypes.c_ushort),
                ('dmFields', ctypes.c_ulong),
                ('dmOrientation', ctypes.c_short),
                ('dmPaperSize', ctypes.c_short),
                ('dmPaperLength', ctypes.c_short),
                ('dmPaperWidth', ctypes.c_short),
                ('dmScale', ctypes.c_short),
                ('dmCopies', ctypes.c_short),
                ('dmDefaultSource', ctypes.c_short),
                ('dmPrintQuality', ctypes.c_short),
                ('dmColor', ctypes.c_short),
                ('dmDuplex', ctypes.c_short),
                ('dmYResolution', ctypes.c_short),
                ('dmTTOption', ctypes.c_short),
                ('dmCollate', ctypes.c_short),
                ('dmFormName', ctypes.c_wchar * 32),
                ('dmLogPixels', ctypes.c_ushort),
                ('dmBitsPerPel', ctypes.c_ulong),
                ('dmPelsWidth', ctypes.c_ulong),
                ('dmPelsHeight', ctypes.c_ulong),
                ('dmDisplayFlags', ctypes.c_ulong),
                ('dmDisplayFrequency', ctypes.c_ulong),
                ('dmICMMethod', ctypes.c_ulong),
                ('dmICMIntent', ctypes.c_ulong),
                ('dmMediaType', ctypes.c_ulong),
                ('dmDitherType', ctypes.c_ulong),
                ('dmReserved1', ctypes.c_ulong),
                ('dmReserved2', ctypes.c_ulong),
                ('dmPanningWidth', ctypes.c_ulong),
                ('dmPanningHeight', ctypes.c_ulong),
                ('dmDisplayOrientation', ctypes.c_ulong),
            ]

        dm = DEVMODE()
        dm.dmSize = ctypes.sizeof(DEVMODE)
        dm.dmFields = 0x00000080

        res = user32.EnumDisplaySettingsW(None, ENUM_CURRENT_SETTINGS, ctypes.byref(dm))
        if res:
            return dm.dmDisplayOrientation
        else:
            return None
            
    except Exception as e:
        print(f"❌ Error getting current rotation: {e}")
        return None

def test_screen_rotation():
    """Test screen rotation functionality"""
    print("🧪 Testing screen rotation...")
    
    # Test getting current rotation
    current = get_current_rotation()
    if current is not None:
        print(f"📺 Current rotation: {current}")
    else:
        print("❌ Could not get current rotation")
        return False
    
    # Test each rotation
    rotations = [DMDO_DEFAULT, DMDO_90, DMDO_180, DMDO_270]
    rotation_names = ["0° (Landscape)", "90° (Portrait)", "180° (Landscape flipped)", "270° (Portrait flipped)"]
    
    for i, rotation in enumerate(rotations):
        print(f"\n🔄 Testing rotation: {rotation_names[i]}")
        success = set_display_rotation(rotation)
        if success:
            print(f"✅ Rotation {rotation} works")
        else:
            print(f"❌ Rotation {rotation} failed")
    
    # Return to default
    print(f"\n🔄 Returning to default rotation...")
    set_display_rotation(DMDO_DEFAULT)
    
    return True

def show_rotation_instructions():
    """Show manual rotation instructions"""
    print("\n" + "="*60)
    print("🔄 MANUAL SCREEN ROTATION INSTRUCTIONS")
    print("="*60)
    print("Since automatic rotation may not work on all systems,")
    print("here are manual methods to rotate your screen:")
    print()
    print("📱 Method 1: Windows Display Settings")
    print("   1. Right-click on desktop → Display settings")
    print("   2. Scroll down to 'Display orientation'")
    print("   3. Select: Landscape, Portrait, Landscape (flipped), or Portrait (flipped)")
    print("   4. Click 'Keep changes' when prompted")
    print()
    print("⌨️  Method 2: Keyboard Shortcuts (if supported)")
    print("   - Ctrl + Alt + → : Rotate 90° right")
    print("   - Ctrl + Alt + ← : Rotate 90° left")
    print("   - Ctrl + Alt + ↓ : Rotate 180°")
    print("   - Ctrl + Alt + ↑ : Return to normal")
    print()
    print("🎮 Method 3: Graphics Driver Control Panel")
    print("   1. Right-click on desktop → Graphics options")
    print("   2. Look for 'Rotation' or 'Orientation' settings")
    print("   3. Select your desired rotation")
    print()
    print("💡 Tips:")
    print("   - Some systems may require administrator privileges")
    print("   - Graphics drivers must support rotation")
    print("   - External monitors may have different rotation options")
    print("   - If rotation doesn't work, try updating graphics drivers")
    print("="*60)
