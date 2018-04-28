# My Web Scraper
# This grew to be much longer than I initially intended...

from selenium import webdriver
import os
from json import dump
from time import sleep


INPUT_LINK = "https://www.congress.gov/search?q={%22source%22:%22legislation%22,%22congress%22:%22109%22,%22type%22:%22bills%22,%22bill-status%22:%22all%22,%22chamber%22:%22Senate%22}"


def get_links(initial_link = INPUT_LINK):
    '''
    Goes through a search on Congress.gov and gathers the URLS for all of the bills.
    Works for any search, but is slow.

    input: input_link, the link generated by a search on congress.gov

    output: 
    links, a list of URLS
    driver, a selenium Chrome webdriver
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
    return links[::2], driver # it gives 2 of each link 

def get_links_hardcoded(number_of_bills = 4122):
    '''
    Hardcoded, but is nearly instantaneous compared to generalized solution (above), which takes a few minutes.
    '''
    list_of_urls = []
    url = 'https://www.congress.gov/bill/109th-congress/senate-bill/?r=1'
    for i in range(1, number_of_bills+1):
        for index, letter in enumerate(url):
            if letter == '?':
                list_of_urls.append(url[:index] + str(i) + url[index:])
                break 
    return list_of_urls

def get_pdf(url, driver):
    '''
    Goes to the pages returned by get_links and downloads the pdfs off of them.

    converts to test files with shell command 'for file in ./*; do pdftotxt $file; done'

    inputs: 
    url, a URL from get_links
    driver, a selenium Chrome webdriver

    output:
    saves all the bills as pdfs
    '''
    for index, letter in enumerate(url):
        if letter == '?':
            url = url[:index] + '/text' + url[index:]
            break 
    print("sleeping...")
    sleep(2.2)
    print("woke up!")
    driver.get(url)
    pdf_link = driver.find_element_by_link_text("PDF").get_attribute('href')
    try:
        _ = driver.find_element_by_class_name("passed")
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

def get_sponsorships(url, driver, counter):
    '''
    Goes to pages returned by get_links and gathers sponsorship/cosponsorship data.
    
    input: url to go to, from get_links

    output: 
    bill_name, a string 'S.####', the bill number
    main_sponsor, a string
    cosponsors, a list of the cosponsors
    '''
    cosponsor_names = []
    for index, letter in enumerate(url):
        if letter == '?':
            url = url[:index] + '/cosponsors' + url[index:]
            break 
    try:
        print("sleeping...")
        sleep(2.2) # Congress.gov/robots.txt prescribes a 2 second delay.
        print("woke up!")
        driver.get(url)
        # try:
        #     bill_name = driver.find_element_by_class_name('legDetail').text.split(' ')[0]
        # except:
        bill_name = 'S.{}'.format(counter)
        sponsor = driver.find_element_by_id('display-message')
        main_sponsor = sponsor.find_element_by_tag_name('a').get_attribute('innerHTML')
        cosponsors = driver.find_elements_by_class_name('actions')
        for cosponsor in cosponsors:
            person = cosponsor.text
            if person[-1] == '*':
                person = person[:-1]
            cosponsor_names.append(person)
    except Exception as E:
        print("Exception!")
        print(E)
        return None, None, None
        
    return bill_name, main_sponsor, cosponsor_names[1:] # cosponsor names has column title -- 'Cosponsors' -- as the 0th element 

def get_all_pdfs(list_of_urls, driver):
    '''
    Downloads all the bills as pdfs

    inputs:
    list_of_urls, list from get_links
    driver, a selenium Chrome webdriver
    '''
    for url in list_of_urls: 
        get_pdf(url, driver)

def get_all_sponsorships(list_of_urls, driver):
    '''
    Gets all the sponsorship/cosponsorship data and saves them as a json

    inputs:
    list_of_urls, list from get_links
    driver, a selenium Chrome webdriver

    outputs:
    a json file of bill-number: (main-sponsor, list of cosponsors)
    '''
    counter = 0
    master_dict = {}
    for url in list_of_urls:
        bill, sponsor, cosponsors = get_sponsorships(url, driver, counter)
        if sponsor == None:
            counter += 1
            continue
        master_dict[bill] = (sponsor, cosponsors)
        print(counter)
        # print("{}: {} with {}".format(bill, sponsor, cosponsors))
        if counter % 100 == 0: # To save progress in case my scraper gets messed with by the nincompoops at Congress.gov.
            print("Saved milestone!")
            with open('milestones/milestone2_{}.json'.format(counter), 'w') as f2:
                dump(master_dict, f2)
        counter += 1
    with open('bill_sponsorships2.json', 'w') as f:
        dump(master_dict, f)


def go(initial_link = INPUT_LINK):
    '''
    Main function; runs the whole thing.

    Input link is the link generated by a search on Congress.gov.  Grabs every bill for any search.

    P.S. Making this more modular made it a heck of a lot easier to write and modify.  I know it could be quicker by 
    only visiting each page once, but the speed boost isn't too much (it reloads the page when I go to cosponsors). 

    '''
    list_of_urls, driver = get_links(initial_link)
    # list_of_urls= get_links_hardcoded(500)
    driver = webdriver.Chrome()
    get_all_pdfs(list_of_urls, driver)
    get_all_sponsorships(list_of_urls, driver)
    driver.close()

if __name__ == "__main__":
    go()



