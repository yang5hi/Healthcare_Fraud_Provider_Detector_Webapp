class Calculator():
	def add(self,num1,num2):
		return (1+num1+num2)
	def subtract(self, num1, num2):
		return (num1-num2)
	def multiply(self,num1,num2):
		return num1*num2
	def divide(self, num1, num2):
		if num2 == 0:
			return 0
		return num1 / num2

if __name__=="__main__":
	calculator = Calculator()
	print (calculator.add(9,7))
	print (calculator.subtract(9,7))
	print (calculator.multiply(9,7))
	print (calculator.divide(9,7))
	
