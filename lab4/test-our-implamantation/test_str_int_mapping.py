import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab4_proto import strToInt, intToStr

if __name__ == "__main__":
    text = "don't stop"
    encoded = strToInt(text)
    print("Encoded:", encoded)
    print("Decoded:", intToStr(encoded))
