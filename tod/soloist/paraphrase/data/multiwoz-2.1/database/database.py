from collections import OrderedDict
import re
import os
import importlib
import json
import types
import shutil
import zipfile
from itertools import chain


MW_DOMAINS = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
DEFAULT_IGNORE_VALUES = ['not mentioned', 'dont care', "don't care", 'dontcare', "do n't care", 'none']


class Database:
    def __init__(self, zipf):
        self.path = zipf.filename
        module = types.ModuleType('convlab_dbquery')
        exec(zipf.read('convlab_dbquery.py').decode('utf-8'), module.__dict__)
        convlab_database = getattr(module, 'Database')
        self.ignore_values = DEFAULT_IGNORE_VALUES
        self.supported_domains = MW_DOMAINS
        self._name_map = None
        self._ontology = None
        self._regexp = None

        # Load database files
        def hacked_init(self):
            self.dbs = {}
            for domain in MW_DOMAINS:
                with zipf.open(f'db/{domain}_db.json') as f:
                    self.dbs[domain] = json.load(f)

        setattr(convlab_database, '__init__', hacked_init)
        self.inner = getattr(module, 'Database')()

        # Load ontology
        if globals().get('DB_ONTOLOGY', True):
            with zipf.open('db_ontology.json') as f:
                self._ontology = {tuple(k.split('-')): set(v) for k, v in json.load(f).items()}
            self._build_replace_dict()

    price_re = re.compile(r'\d+\.\d+')

    @staticmethod
    def hack_query(belief):
        new_belief = OrderedDict()
        for domain, bs in belief.items():
            new_bs = OrderedDict()
            new_belief[domain] = new_bs
            for key, val in bs.items():
                val = bs[key]
                if domain == 'restaurant' and key == 'name' and val.lower() == 'charlie':
                    val = 'charlie chan'
                if domain == 'restaurant' and key == 'name' and val.lower() == 'good luck':
                    val = 'the good luck chinese food takeaway'
                # if domain == 'hotel' and key == 'name' and val.lower() == 'el shaddai guesthouse':
                #     val = 'el shaddai'
                new_bs[key] = val
        return new_belief

    @staticmethod
    def capitalize(val):
        def _mk(v):
            i, v = v
            if i == 0 or v not in {'the', 'an', 'a', 'of', 'in', 'for', 'as', 'these', 'at', 'up', 'on', 'and', 'or'}:
                return v[:1].upper() + v[1:]
            else:
                return v
        return ' '.join(map(_mk, enumerate(val.split())))

    @staticmethod
    def map_database_key(key):
        if key == 'trainID':
            key = 'id'
        key = ''.join([' '+i.lower() if i.isupper()
                       else i for i in key]).lstrip(' ')
        key = key.replace('_', ' ')
        if key == 'pricerange':
            key = 'price range'
        if key == 'taxi phone' or key == 'phone':
            key = 'phone'
        if key == 'taxi colors':
            key = 'color'
        if key == 'taxi types':
            key = 'brand'
        if key == 'ref':
            key = 'reference'
        if key == 'leaveAt':
            key = 'leave at'
        if key == 'arriveBy':
            key = 'arrive by'
        if key == 'entrance fee':
            key = 'fee'
        return key

    @staticmethod
    def map_database_row(domain, row, query):
        results = dict()
        for k, val in row.items():
            k2 = Database.map_database_key(k)
            if k == 'location':
                continue
            elif k == 'post code' or k == 'postcode':
                val = val.upper()
            elif k == 'name':
                val = Database.capitalize(val)
            elif k == 'type' and val == 'concerthall':
                val = 'concert hall'
            elif k == 'price' and domain == 'hotel' and isinstance(val, dict):
                val = val.get('single', val.get('double', next(iter(val.values()))))
                val = f'{val} pounds'
            if k2 == 'people':
                # BUG in MW2.0
                val = val.lstrip('`')
            results[k2] = val
        if 'color' in results and 'brand' in results:
            results['car'] = f"{results['color']} {results['brand']}"
        if domain == 'train' and 'price' in row and 'people' in query:
            people = int(query['people'])

            def multiply_people(m):
                price = float(m.group(0))
                price *= people
                return format(price, '.2f')
            if people != 1:
                results['price'] = Database.price_re.sub(multiply_people, row['price'])
        return results

    @staticmethod
    def normalize_for_db(s):
        s = ','.join(s.split(' ,'))
        s = s.replace('swimming pool', 'swimmingpool')
        s = s.replace('night club', 'nightclub')
        s = s.replace('concert hall', 'concerthall')
        return s

    @staticmethod
    def translate_to_db_col(s):
        if s == 'leave at':
            return 'leaveAt'
        elif s == 'arrive by':
            return 'arriveBy'
        elif s == 'price range':
            return 'pricerange'
        else:
            return s

    def domain_not_empty(self, domain_bs):
        return any(len(val) > 0 and val not in self.ignore_values for val in domain_bs.values())

    def _build_replace_dict(self):
        if self._regexp is not None:
            return
        clear_values = {'the', 'a', 'an', 'food'}
        clear_values.update(self._ontology[('hotel', 'type')])
        clear_values.update(self._ontology[('hotel', 'price range')])
        clear_values.update(self._ontology[('hotel', 'area')])
        clear_values.update(self._ontology[('restaurant', 'price range')])
        clear_values.update(self._ontology[('restaurant', 'food')])
        clear_values.update(self._ontology[('restaurant', 'area')])
        clear_values.update(self._ontology[('attraction', 'type')])
        clear_values = (f' {x} ' for x in clear_values)
        self._regexp = re.compile('|'.join(map(re.escape, clear_values)))
        db_entities = chain(self.inner.dbs['attraction'], self.inner.dbs['hotel'], self.inner.dbs['restaurant'])
        self._name_map = {self._clear_name(r): r['name'].lower() for r in db_entities}

    def _clear_name(self, domain_bs):
        name = ' ' + domain_bs['name'].lower() + ' '
        name = self._regexp.sub(' ', name)
        name = re.sub(r'\s+', ' ', name)
        name = name.strip()
        return name

    @staticmethod
    def _to_minutes(time):
        hour, minutes = tuple(map(int, time.split(':')))
        return minutes + 60 * hour

    def __call__(self, belief, return_results=False):
        belief = Database.hack_query(belief)
        all_results = OrderedDict()
        for domain, domain_bs in belief.items():
            if domain not in self.supported_domains:
                continue  # skip unsupported domains
            if self.domain_not_empty(domain_bs) or \
                    domain in [d.lower() for d in {'Police', 'Hospital'}]:
                def query_single(domain_bs):
                    blocked_slots = {'people', 'booked', 'stay'}
                    if domain != 'train' and domain != 'bus':
                        blocked_slots.add('day')
                    query_bs = [(Database.translate_to_db_col(slot), Database.normalize_for_db(val))
                                for slot, val in domain_bs.items() if slot not in blocked_slots]
                    result = self.inner.query(domain, query_bs)
                    result = [Database.map_database_row(domain, k, domain_bs) for k in result]

                    # Implement sorting missing in convlab
                    if domain == 'train' and 'arrive by' in domain_bs:
                        result.sort(key=lambda x: self._to_minutes(x['arrive by']), reverse=True)
                    elif domain == 'train' and 'leave at' in domain_bs:
                        result.sort(key=lambda x: self._to_minutes(x['leave at']))
                    return result
                result = query_single(domain_bs)
                if len(result) == 0 and 'name' in domain_bs and self._clear_name(domain_bs) in self._name_map:
                    domain_bs = dict(**domain_bs)
                    domain_bs['name'] = self._name_map[self._clear_name(domain_bs)]
                    result = query_single(domain_bs)

                if return_results:
                    all_results[domain] = (len(result), result)
                else:
                    all_results[domain] = len(result)
        return all_results

    def save(self, path):
        shutil.copy(self.path, os.path.join(path, os.path.split(self.path)[-1]))
