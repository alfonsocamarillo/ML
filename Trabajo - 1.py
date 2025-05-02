import pandas as pd
import os
import shutil
import time
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
import json
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.microsoft import EdgeChromiumDriverManager

ruta_descargas=os.getcwd()
options = Options()
options.add_argument("--disable-notifications")
options.add_argument("--disable-infobars")
options.add_argument("--mute-audio")
options.add_argument("--start-maximized")
#options.add_argument("--headless")
prefs = {
    "download.default_directory": ruta_descargas,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
}
options.add_experimental_option("prefs", prefs)
service = Service(EdgeChromiumDriverManager().install())
#service = Service(driver_path) # ruta donde este su driver.exe instalado a mano
driver = webdriver.Edge(service=service, options=options)

# Ejercicio 1
buscar = "analista de datos"
ubicacion = "mexico"
driver.get("https://mx.indeed.com/")
time.sleep(5)
driver.find_element(By.XPATH,"//input[@id='text-input-what']").send_keys(buscar)
driver.find_element(By.XPATH,"//input[@id='text-input-where']").send_keys(ubicacion)
driver.find_element(By.XPATH,"//input[@id='text-input-what']").send_keys(Keys.ENTER)
time.sleep(5)

empleo=[]
empresa=[]
locacion=[]
info1=[]
info2=[]
tiempo=[]

a=driver.find_elements(By.XPATH,"//a/span")
b=driver.find_elements(By.XPATH,"//div[@data-testid='timing-attribute']/span")
c=driver.find_elements(By.XPATH,"//div[@data-testid='timing-attribute']/div")
d=driver.find_elements(By.XPATH,"//div[@data-testid='attribute_snippet_testid']")
e=driver.find_elements(By.XPATH,"//div[@class='underShelfFooter']//li")
f=driver.find_elements(By.XPATH,"//span[@data-testid='myJobsStateDate']")

for i in range(1,len(a)):
    elemento=a[i].text.split("\n")
    empleo.append(elemento[0])
   # vistas.append(elemento[1])
   # tiempo.append(elemento[2])

for i in range(len(b)):
    empresa.append(b[i].text)
    
for i in range(len(c)):
    locacion.append(c[i].text)
    
for i in range(len(d)):
    info1.append(d[i].text)
    
for i in range(len(e)):
    info2.append(e[i].text)
    
for i in range(len(f)):
    tiempo.append(f[i].text)

data = list(zip(empleo,empresa,locacion,info1,info2,tiempo))
columns = ["Titulo empleo","Empresa","Ubicacion","Informacion","Habilidades","Activo"]
data = pd.DataFrame(data,columns=columns)

driver.quit()








