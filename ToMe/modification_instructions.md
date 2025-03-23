# Instructions for Modifying ToMe

This document provides step-by-step instructions to manually modify the ToMe library for your specific use case.
Ensure that you have cloned the original repository as described in the [README.md](./README.md) file.

---

## 変更手順
#### 1. Open tome/patch/timm.py in ToMe repository
#### 2. Locate the following sections in the file
The class : class ToMeBlock
The function : def make_tome_class
The function : def apply_patch
#### 3. Replace the existing code in these sections with the code below:

```python
# modified code of class ToMeBlock
class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r=112　#Assign the number of tokens to merge

        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
                

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


# modified code of make_tome_class
def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwdargs) -> torch.Tensor:
            self._tome_info["r"] = self.r
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwdargs)

    return ToMeVisionTransformer

# Custom implementation for apply_patch
def apply_patch(
    model: VisionTransformer, changeLayersNum: int = 1,trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = 1
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": False,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True
    for module in model.modules():
        if isinstance(module, Block):
            changeLayersNum-=1
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info

        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
            if changeLayersNum == 0:
                break
```
## Usage Notes
Except for the additional arguments added to apply_patch, the usage of ToMe remains the same as the original implementation.
Refer to the original ToMe documentation for further details on its usage.

## Additional Notes
These changes implement ToMe's functionality with additional features, such as adjustable token merging.
Modifications to timm are also required. Ensure that you follow the instructions in the timm/ directory to modify the necessary files.
