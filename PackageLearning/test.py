import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="pastel", context="paper")

products = ["Product A", "Product B", "Product C", "Product D"]
sales = [120, 210, 150, 180]

sns.barplot(x=products, y=sales)

# 添加标签和标题
plt.xlabel("Products")
plt.ylabel("Sales")
plt.title("Product Sales by Category")

# 显示图表
plt.show()