def main(): 
    print("This program converts Celsius to Fahrenheit.")
    celsius = eval(input("What is the Celsius temperature? "))
    fahrenheit = 9/5 * celsius + 32
    print("The temperature is", fahrenheit, "degrees Fahrenheit.")

if __name__ == '__main__':
    main()