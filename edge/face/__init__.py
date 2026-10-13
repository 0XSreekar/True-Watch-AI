"""Face detection (YuNet, pretrained) and privacy (blur-by-default, access-controlled unsealing).

See docs/ARCHITECTURE_V2.md section 1.1 ("face data stays on the post, 30-day
retention, unsealing is access-controlled") and `privacy.py`'s module
docstring for how that is enforced.
"""

from __future__ import annotations
