from selenium import webdriver
import time

driver = webdriver.Chrome(executable_path=r"C:\Program Files (x86)\Google\chromedriver.exe")
#driver.get(r'./testmap.html')
driver.get(r'file:///C:/Users/sstalley/Documents/code/CuP/testmap.html')
time.sleep(5)
driver.save_screenshot(r"./Figures/screenshot.png")
driver.close()
