from app.src.calculator import Calculator

cal = Calculator()

def test_add():
	assert cal.add(1,1) == 2
	assert cal.add(2,2) == 4

def test_subtract():
	assert cal.subtract(2,2) == 0
	assert cal.subtract(5,2) == 3
