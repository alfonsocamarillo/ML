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

driver.get("https://pokemon.fandom.com/es/wiki/Lista_de_Pokémon")
time.sleep(10)

n_poke=[]
name_poke=[]
name_poke1=[]
pokemon=[]
tipo1=[]
tipo1_1=[]

a=driver.find_elements(By.XPATH,"//table[@class='tabpokemon sortable mergetable jquery-tablesorter']//td[1]")
b=driver.find_elements(By.XPATH,"//table[@class='tabpokemon sortable mergetable jquery-tablesorter']//td[2]/a")
c=driver.find_elements(By.XPATH,"//table[@class='tabpokemon sortable mergetable jquery-tablesorter']//td[3]//span")
cc=driver.find_elements(By.XPATH,"//table[@class='tabpokemon sortable mergetable jquery-tablesorter']//td[4]")
aa=driver.find_elements(By.XPATH,"//table[@class='tabpokemon sortable mergetable jquery-tablesorter']//td[3]/a")

for i in range(len(a)):
    n_poke.append(a[i].text)
    
    
for i in range(0,721):
    name_poke.append(b[i].text)    

for i in range(len(aa)):
    name_poke1.append(aa[i].text)    

pokemon=name_poke+name_poke1

for i in range(len(c)):
    tipo1.append(c[i].text)    

for i in range(len(c),len(cc)):
    tipo1_1.append(cc[i].text)    
tipo1_c=tipo1+tipo1_1

    
data_poke = list(zip(n_poke,pokemon,tipo1_c))
columns = ["#","Pokemon","Tipo 1"]
data_poke = pd.DataFrame(data_poke,columns=columns)

driver.quit()









