import requests
from bs4 import BeautifulSoup

def fetch_and_extract(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content_div = soup.find('div', class_='mw-parser-output')
            if not content_div:
                print("The expected content container was not found.")
                return ""
            
            paragraphs = content_div.find_all('p')
            extracted_text = '\n\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

            with open('Selected_Document.txt', 'w', encoding='utf-8') as file:
                file.write(extracted_text)

            print("Page successfully retrieved and content saved to 'Selected_Document.txt'.")
            return extracted_text
        else:
            print(f"Failed to retrieve the page. HTTP Status Code: {response.status_code}")
            return ""
    except requests.RequestException as e:
        print(f"An error occurred while fetching the URL: {e}")
        return ""

def main():
    # Hardcoded URL here:
    url = "https://en.wikipedia.org/wiki/Seahorse"  
    fetch_and_extract(url)

if __name__ == '__main__':
    main()


