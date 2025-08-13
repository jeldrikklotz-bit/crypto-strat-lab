# test_backend.py
import os
os.environ.setdefault("MPLBACKEND","TkAgg")  # or QtAgg
import matplotlib.pyplot as plt
plt.ion()
fig, ax = plt.subplots()
ax.plot([0,1,2],[0,1,0])
ax.set_title("Backend OK")
plt.show(block=True)
