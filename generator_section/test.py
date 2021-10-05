def test():
	s = input("Введите")
	b = ''
	strok = ''
	n = 0
	for i in s:
	  try:
	    i = int(i)
	    i = str(i)
	    b = b + i
	  except ValueError:
	    if (b != ''):
	      n = int(b)
	      b = ''
	      while n > 0:
	        b = str(n % 2) + b
	        n = n // 2
	      strok = strok + b
	      b = ''
	  if (b == ''):
	    strok = strok + i
	if (b != ''):
	 n = int(b)
	 b = ''
	 while n > 0:
	     b = str(n % 2) + b
	     n = n // 2
	 strok = strok + b
	 b = ''
	print(strok)


alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# r1 = int(input("Введите число: "))
# r2 = int(input("Введите число системы счисления: "))

r1 = 1
r2 = 10


def from_ten_to_p_drobi(num, base):
    changed_d = [] * 5
    num = num // 100
    kontrol_of_cicle = num
    logikp = 1
    i = 1
    while(num != 1) or (logikp != 0):
        num = num*base
        c = num
        c = int(c)
        c = round(c)
        if c >= 10:
            changed_d[i+1] = alphabet[c-10]
        i = i+1
        num = num-c
        if kontrol_of_cicle == num:
            logikp = 0
    return (changed_d)


print(from_ten_to_p_drobi(r1, r2))


if __name__ == '__main__':
    print(from_ten_to_p_drobi(r1, r2))
