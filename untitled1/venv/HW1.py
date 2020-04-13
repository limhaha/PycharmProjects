from math import *


def is_prime(n):
    if n != 1:

        for i in range(2, floor(sqrt(n))):

            if (n % i) == 0:
                print(n, "can divide by", i)
                return False

    else:
        return False

    return True


num = input("Enter number: ")

if is_prime(int(num)):
    print(num + " is prime number.")

else:
    print(num + " is not prime number.")

