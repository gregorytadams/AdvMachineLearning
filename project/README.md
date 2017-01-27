# TeamDragon-Project
Group Project for CS122 based on the GovTrack Data and API.

Set Up Instructions (from scratch with no database or JSON files, which are now included):

1. Run "python manage.py makemigrations" and then "python manage.py migrate" in order to create the database
2. Run "python database_setup.py" from /mysite in order to grab the data from GovTrack (This will take a long time)
3. Run "python leadership_score.py" in order to add leadership scores and cosponsor counts to the database
4. Install networkx with "pip3 install networkx"
5. Run "python create_json_graph.py" in order to generate JSON files for the network graphs
6. Run "python create_party_json.py" in order to generate specific network JSON files for each party

To view the website:

1. Go into mysite directory
2. run "python3 manage.py runserver"
3. Copy/paste the address given into a browser.


Libraries Necessary:
Django 1.8 (Python)
NetworkX (Python)
D3.js
plotly.js