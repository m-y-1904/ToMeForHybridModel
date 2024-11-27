# Instructions for Modifying ToMe

This document provides step-by-step instructions to manually modify the ToMe library for your specific use case.
Ensure that you have cloned the original repository as described in the [README.md](./README.md) file.

---

## 変更手順
#### 1. Open tome/patch/timm.py in ToMe repository
#### 2. Locate the following sections in the file
The class : class ToMe
The function : def make_tome_class
The function : def apply_patch
#### 3. Replace the existing code in these sections with the code below:

```python
# Custom implementation for class ToMe
class ToMe:
    # Your custom implementation here
    def __init__(self, param):
        self.param = param
    
    def merge_tokens(self, tokens):
        # Example merging logic
        pass

# Custom implementation for make_tome_class
def make_tome_class():
    # Your custom implementation here
    pass

# Custom implementation for apply_patch
def apply_patch():
    # Your custom implementation here
    pass
```
