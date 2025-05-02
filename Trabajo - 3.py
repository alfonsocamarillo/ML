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


buscar="Calculo"
driver.get("https://porrua.mx/")
time.sleep(2)

driver.find_element(By.XPATH,"//input[@id='search']").send_keys(buscar)
driver.find_element(By.XPATH,"//input[@id='search']").send_keys(Keys.ENTER)

time.sleep(5)


nombre=[]
imagen=[]
autor=[]
precio=[]


a=driver.find_elements(By.XPATH,"//strong[@class='product name product-item-name']/a")
b=driver.find_elements(By.XPATH,"//div[@class='author']//a")
c=driver.find_elements(By.XPATH,"//span[@class='product-image-wrapper']/img")
d=driver.find_elements(By.XPATH,"//span[@class='price']")


for i in range(len(a)):
    nombre.append(a[i].text)
    
    
for i in range(len(b)):
    autor.append(b[i].text)    
    
for i in range(len(c)):
    imagen.append(c[i].get_attribute("src"))    
    
for i in range(len(d)):
    precio.append(d[i].text)    

data_libro = list(zip(nombre,autor,precio,imagen))
columns = ["Nombre del libro","Autor","Precio Actual","Portada"]
data_libro = pd.DataFrame(data_libro,columns=columns)

driver.quit()









