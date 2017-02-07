#My Web Scraper

import csv
from selenium import webdriver
import time


INPUT_LINK = "https://www.congress.gov/search?q={%22source%22:%22legislation%22,%22congress%22:%22109%22,%22type%22:%22bills%22,%22bill-status%22:%22all%22,%22chamber%22:%22Senate%22}"


def get_links(search_link = INPUT_LINK):
    '''
    Spider/crawler thingy
    '''
    links = []
    driver = webdriver.Chrome()
    driver.get(search_link)
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
    return links[::2] #it's giving me 2 of each link.  Tbh, this is easier than figuring out why... 


def get_bill_text(url):
    for index, letter in enumerate(url):
        if letter == '?':
            url = url[:index] + '/text' + url[index:]
    return url
    # driver.get(url)

    #     html = c.read().decode('utf-8')
    #     soup = BS(html, 'html.parser')
    # pass

def save_bill_text(bill_text, file_location_name):
    with open(file_location_name, 'w') as f:
        f.write(bill_text)
    print("saved {}".format(file_location_name))

def go(initial_link):
    counter = 0
    list_of_urls = get_links(INPUT_LINK)
    for url in list_of_urls: #for some reason I really can't figure out, it's giving me 2 of each link.  This is easier.
        bill_text = get_bill_text(ur)
        save_bill_text(bill_text, "law" + str(counter))
    print("saved {} laws".format(counter))
    #REMEMBER TO driver.close()






def fetch_data(url):
    '''
    Not completed, but prints all the relevant info.  Just need to format it in a reasonable way.
    
    Output: list of lists
    '''
    with urllib.request.urlopen(url) as c:
        html = c.read().decode('utf-8')
        soup = BS(html, 'html.parser')
        for table in soup.find_all('table'):
            try:
                if table.get('class')[0] == "collapsible":
                    print("in if block")             
                    for tr in table.find("tr"):
                        print("started tr loop")
                        # print(type(tr))
                        # print(tr)
                        heading = [th.get_text() for th in table.find("tr").find_all("th")]
                        print("got past heading")
                        print(tr.find("td"))
                        for td in tr.find("td"):
                            print("in td loop")
            except Exception as e:
                print(e)

    return "Hi!"
                        # print(nl)
                    # headings = [th.get_text() for th in table.find("tr").find_all("th")]


                    # print(headings)
            # except:
            #     # print("this", table.get('class'))
            #     continue
    # return None

def parse_text(text):
    for line in text:
        print(len(line))
        print(line)
        print('-'*300)
    pass

def generate_links(dict_of_states, years):
    '''
    '''
    rv = []
    for abbrev in dict_of_states:
        for year in years:
            s = "https://ballotpedia.org/United_States_Senate_election_in_{},_{}".format(dict_of_states[abbrev], year)
            s2 = "https://ballotpedia.org/United_States_Senate_elections_in_{},_{}".format(dict_of_states[abbrev], year)
            rv.append(s)
            rv.append(s2)
    return rv


def go():
    output_list = []
    list_of_links = generate_links(STATES, YEARS)
    for link in list_of_links:
        try:
            output_list += fetch_data(link)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                print("page {} does not exist; there was no election".format(link))
            else:
                print("If you see this the script is fucked, sorry")
    # with open(OUTPUT_NAME) as output:
    #     writer = csv.writer(output)
    #     for line in rv:
    #         writer.writerow(line)
    # print("Created file {}".format(OUTPUT_NAME))

# if __name__ == "__main__":
#     go()

