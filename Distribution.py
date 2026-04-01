import os

import sys

import matplotlib.pyplot as plt

data = {}
pourcentages = {}

def get_the_bar_chart(directoryName):
    plt.figure(figsize=(15, 6))
    colors = plt.cm.tab20.colors
    plt.bar(data.keys(), data.values(), color=colors[:len(data)])
    plt.title(directoryName)
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.tight_layout()

def get_the_pie_chart(directoryName):
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab20.colors
    plt.pie(pourcentages.values(), labels=pourcentages.keys(), autopct='%1.1f%%', colors=colors[:len(pourcentages)])
    plt.title(directoryName)
    plt.axis('equal')
    plt.tight_layout()

def main():
    if(len(sys.argv) != 2):
        print("Usage: python3 Distribution.py <filename>")
        return
    directoryName = sys.argv[1]
    for className in os.listdir(directoryName):
        classPath = os.path.join(directoryName, className)
        if os.path.isdir(classPath):
            images = [
                f for f in os.listdir(classPath)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            data[className] = len(images)
    count_the_total = sum(data.values())
    for each in data:
        pourcentages[each] = (data[each] / count_the_total) * 100
    get_the_bar_chart(sys.argv[1])
    get_the_pie_chart(sys.argv[1])
    plt.show()
main()