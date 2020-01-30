import matplotlib.pyplot as plt

x1 = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.2, 0.3]
y1 = [90.51, 90.72, 90.73, 90.67, 90.51, 90.20, 89.89, 88.64, 87.77]

# x2 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# y2 = []

# fig= plt.figure()

# fig, ax = plt.subplots(1,2)
# print(ax)
# ax[0].plot(x1, y1)
# fig.show()

plt.plot(x1, y1)
plt.ylabel('mAP(%)')
plt.xlabel('Threshold T')
plt.title('The influence of Threshold T')
plt.show()