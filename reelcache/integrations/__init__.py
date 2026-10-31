"""Optional integrations of the ReelCache runtime with serving engines.

Nothing in :mod:`reelcache` core imports this package; each integration imports its
engine only where the engine calls into it (never at ReelCache import time).
"""
