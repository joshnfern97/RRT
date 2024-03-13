# Importing necessary libraries
import pil
import sys
sys.modules['PIL'] = pil
import matplotlib.pyplot as plt
import pandas as pd

# Defining functions or classes

def main():
    # Main function where the execution begins
    df = pd.read_csv('obstacles.csv')

    print(df.to_string()) 
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    plt.plot(x,y, 'ro')
    plt.plot(0,0, 'go')
    plt.plot(100,100, 'gx')
    plt.show()

if __name__ == "__main__":
    # Calling the main function if the script is executed directly
    main()