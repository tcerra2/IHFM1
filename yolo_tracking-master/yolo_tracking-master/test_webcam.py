#!/usr/bin/env python
"""Test script to run tracking with webcam"""

import sys
import time

# Add project to path
sys.path.insert(0, str(__file__.split('test_webcam')[0]))

from tracking.track import parse_opt, run

if __name__ == "__main__":
    # Parse arguments
    opt = parse_opt()
    opt.source = '0'
    opt.show = True
    opt.conf = 0.4
    
    print("=" * 60)
    print("YOLO Tracking Webcam Test")
    print("=" * 60)
    print(f"Source: {opt.source} (Webcam)")
    print(f"Show: {opt.show}")
    print(f"Confidence: {opt.conf}")
    print("=" * 60)
    print("\nStarting tracking... Press 'q' or spacebar to quit")
    print("=" * 60 + "\n")
    
    try:
        run(opt)
    except KeyboardInterrupt:
        print("\n\nTracking stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
