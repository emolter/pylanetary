#!/usr/bin/env python

'''
This script builds the data in this directory.

It queries the Planetary Ring Node for static (non-ephemeris) ring data
     e.g. https://pds-rings.seti.org/uranus/uranus_rings_table.html for Uranus
It parses the html
It writes the useful data as an hdf5 file, readable by astropy table
    e.g. table.Table.read('Uranus_ring_data.hdf5', format = 'hdf5')
It also writes the references for those values as a text file.
Those numbers should correspond to the numbers in the "footnotes" column
of the ring data tables.

Doubtful these values will change much until the next Uranus/Neptune mission
So no need to rebuild the tables very often

Last run 05/14/22 by emolter
'''

from astropy import table
from bs4 import BeautifulSoup
import numpy as np
import urllib.request

planet_defaults = {
    'jupiter': {'dtypes': [str, float, float, float, float, str, str, str, str],
                },
    'saturn': {'dtypes': [str, float, float, float, str, str, str, str],
               },
    'uranus': {'dtypes': [str, float, float, float, float, float, str, str, str, str],
               },
    'neptune': {'dtypes': [str, float, float, float, str, str, str, str],
                },
}


def parse_ring_data(planet):
    '''
    read in html ring data from planetary ring node
    e.g. https://pds-rings.seti.org/uranus/uranus_rings_table.html
    parse it
    return astropy table of useful quantities
    '''

    URL = f"https://pds-rings.seti.org/{planet.lower()}/{planet.lower()}_rings_table.html"

    with urllib.request.urlopen(URL) as response:
        src = response.read()

    # with open(f'{planet.lower()}_ring_data.html') as f:
    #    src = f.read()
    soup = BeautifulSoup(src, "html.parser")

    # parse the ring data
    tfull = soup.table
    thead = tfull.thead
    headvals = thead.find_all('th')
    headvals = [ele.text.strip() for ele in headvals]

    t = table.Table(names=headvals,
                    dtype=planet_defaults[planet.lower()]['dtypes'])
    nfloats = np.sum(np.asarray(
        planet_defaults[planet.lower()]['dtypes']) == float)

    tbody = tfull.tbody
    rows = tbody.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        cols[1:1 + nfloats] = [s.replace(',', '').strip('~<>')
                               if s != '' else '0' for s in cols[1:1 + nfloats]]
        cols[1:1 + nfloats] = [np.nan if s.upper().isupper() else s.replace('^',
                                                                            'e').replace('0-', '0e-') for s in cols[1:1 + nfloats]]
        t.add_row(cols)

    # parse the references
    refs = soup.find_all(id=lambda x: x and x.startswith('source-'))

    return t, refs


if __name__ == "__main__":

    for planet in ['Jupiter', 'Saturn', 'Uranus', 'Neptune']:
        t, refs = parse_ring_data(planet)
        t.write(f'{planet}_ring_data.hdf5', format='hdf5', overwrite=True)

        with open(f'{planet}_refs.txt', 'w') as f:
            for i, ref in enumerate(refs):
                f.write(str(i + 1) + '. ' + ref.text)
                f.write('\n')
