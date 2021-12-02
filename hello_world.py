def main(): 
    print("This program converts Celsius to Fahrenheit.")
    celsius = eval(input("What is the Celsius temperature? "))
    fahrenheit = 2 * celsius + 30
    print("The temperature is", fahrenheit, "degrees Fahrenheit.")

if __name__ == '__main__':
    main()