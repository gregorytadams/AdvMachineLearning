#My Web Scraper

from selenium import webdriver
import os


INPUT_LINK = "https://www.congress.gov/search?q={%22source%22:%22legislation%22,%22congress%22:%22109%22,%22type%22:%22bills%22,%22bill-status%22:%22all%22,%22chamber%22:%22Senate%22}"


def get_links(initial_link = INPUT_LINK):
    '''
    Spider/crawler thingy
    '''
    links = []
    driver = webdriver.Chrome()
    driver.get(initial_link)
    while True: 
        for span in driver.find_elements_by_class_name("result-heading"):
            a = span.find_element_by_tag_name("a")
            links.append(a.get_attribute('href'))
        try:
            new_page = driver.find_element_by_class_name('next').get_attribute('href')
            driver.get(new_page)
            continue
        except:
            break
    return links[::2], driver #it's giving me 2 of each link.  Tbh, this is easier than figuring out why... 


def get_pdf(url, driver):
    for index, letter in enumerate(url):
        if letter == '?':
            url = url[:index] + '/text' + url[index:]
            break 
    driver.get(url)
    pdf_link = driver.find_element_by_link_text("PDF").get_attribute('href')
    try:
        a = driver.find_element_by_class_name("passed")
        folder = "./passed/"
    except:
        folder = "./not_passed/"
    try: 
        os.chdir(folder)
        os.system('wget {}'.format(pdf_link))
        os.chdir('..')
        print("Downloaded {}".format(pdf_link))
    except Exception as E:
        print(E)
    

def go(initial_link = INPUT_LINK):
    counter = 0
    list_of_urls, driver = get_links(initial_link)
    for url in list_of_urls: 
        get_pdf(url, driver)
        counter += 1
        print("downloaded {} laws".format(counter))
    driver.close()


if __name__ == "__main__":
    go()


