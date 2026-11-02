import os
import cv2
import numpy as np
import time
import argparse
from datetime import datetime
import json

class ColorGenerator:
    def __init__(
        self, 
        screen_width=1920, 
        screen_height=1080, 
        color_interval=1.0
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.color_interval = color_interval
        
        self.colors = [
            (255, 0, 0),     # Blue
            (0, 255, 0),     # Green
            (0, 0, 255),     # Red
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Yellow
        ]
        
        self.color_names = ["Blue", "Green", "Red", "Magenta", "Yellow"]
        self.current_color_index = 0
        self.start_time = time.time()
        self.color_changes = []
        
    def generate_screen(self):
        color = self.colors[self.current_color_index]
        screen = np.full((self.screen_height, self.screen_width, 3), color, dtype=np.uint8)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        cv2.putText(screen, f"Time: {timestamp}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(screen, f"Press 'q' to quit", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return screen
    
    def run(self, fullscreen=True):
        window_name = "Color Sync Generator"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        
        if fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("Starting color generator...")
        print("Colors will change every {:.1f} seconds".format(self.color_interval))
        print("Press 'q' to quit")
        
        last_change_time = time.time()
        
        try:
            while True:
                current_time = time.time()
                if current_time - last_change_time >= self.color_interval:
                    timestamp = time.time()
                    color_name = self.color_names[self.current_color_index]
                    self.color_changes.append({
                        "timestamp": timestamp,
                        "color_index": self.current_color_index,
                        "color_name": color_name,
                    })
                    
                    print(f"[{timestamp}] Changed to {color_name}")
                    self.current_color_index = (self.current_color_index + 1) % len(self.colors)
                    last_change_time = current_time
                
                screen = self.generate_screen()
                cv2.imshow(window_name, screen)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            self.save_log()
    
    def save_log(self):
        """Save color change log to file"""
        filename = os.path.join(
            os.path.dirname(__file__), 
            f"color_sync_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filename, 'w') as f:
            json.dump(self.color_changes, f, indent=4)
        print(f"\nColor change log saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Color Synchronization Generator")
    parser.add_argument("--width", type=int, default=1920, help="Screen width")
    parser.add_argument("--height", type=int, default=1080, help="Screen height")
    parser.add_argument("--interval", type=float, default=0.4, 
                       help="Color change interval in seconds")
    parser.add_argument("--windowed", action="store_true", 
                       help="Run in windowed mode instead of fullscreen")
    
    args = parser.parse_args()
    
    generator = ColorGenerator(
        screen_width=args.width,
        screen_height=args.height,
        color_interval=args.interval
    )
    
    generator.run(fullscreen=not args.windowed)