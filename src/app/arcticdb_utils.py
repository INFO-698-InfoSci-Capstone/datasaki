import arcticdb
import json
from typing import Dict
from app.database import get_db
from app.utils import get_column_stats, NpEncoder, make_list
from app import schemas, crud, models
from app.datasaki_ai_1 import DatasakiAIClient
import weakref
from app.datasaki_ai.query import get_industry, get_feature_details, get_industry_by_msg

class ArcticDBManager:
    _arctic_stores: Dict[int, arcticdb.Arctic] = {}

    @classmethod
    def get_arctic_store(cls, tenant_id: int):
        """
        Get or create an ArcticDB store for a tenant.

        :param tenant_id: Tenant ID to differentiate namespaces
        :return: ArcticDB store for the tenant
        """
        if tenant_id in cls._arctic_stores:
            return cls._arctic_stores[tenant_id]

        # Create a new ArcticDB instance for the tenant
        uri = f"lmdb://data/arctic_lmdb/tenant_{tenant_id}/"
        store = arcticdb.Arctic(uri)

        # Cache the instance to avoid reopening it
        cls._arctic_stores[tenant_id] = store
        return store

class ArcticDBInstance:
    _instances = weakref.WeakValueDictionary()

    def __new__(cls, tenant_id, dataset_id=None,dataset_name: str = None, *args, **kwargs):
        key = (tenant_id, dataset_id)
        if key not in cls._instances:
            instance = super(ArcticDBInstance, cls).__new__(cls)
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, tenant_id: int, data=None, dataset_id: int = None, dataset_name: str = None,
                 create_if_missing: bool = False, **kwargs):
        if not hasattr(self, '_initialized'):
            self._tenant_id = tenant_id
            self._store = ArcticDBManager.get_arctic_store(tenant_id=tenant_id)
            self.aiclient = DatasakiAIClient(tenant_id)
            self._industry = None
            if data is not None:
                library = self._store.create_library(dataset_name)
                self._dtypes = json.dumps(get_column_stats(data), cls=NpEncoder)
                self._industry = self.get_dataset_industry_from_ai()
                library.write(dataset_name, data=data,
                              metadata={"dtypes": self._dtypes,
                                        "history": [], "settings": {}, "context_variables": {},"industry":self._industry,"target":None})
                dataset_data = schemas.DatasetCreate(
                    name=dataset_name,
                    description=kwargs.get('description', ""),
                    tenant_id=tenant_id
                )
                db = next(get_db())
                db_dataset = crud.create_dataset(db, dataset_data)
                dataset_id = db_dataset.id

            self._dataset_id = dataset_id if dataset_id else self.get_dataset_id_from_name(dataset_name, tenant_id=tenant_id)
            self._dataset_name = dataset_name if dataset_name else self.get_dataset_name_from_id(dataset_id)
            self._library = self._store.get_library(self._dataset_name, create_if_missing=create_if_missing)
            self._dtypes = self.get_details('dtypes')
            self._industry = self.get_details('industry')
            self._history = self.get_details('history')
            self._settings = self.get_details('settings')
            self._context_variables = self.get_details('context_variables')
            self._original_dataset_dim = self.get_details('dataset_dim')
            self._original_rows = self.get_details('rows')
            self._initialized = True
            self._target = self.get_details('target')
            self._relationship  = self.get_details('relationship')
            self.update_feature_details()

    @staticmethod
    def update_dtypes_with_feature_information(data1,data2):
        if 'features' not in data2.keys():
            print(data2)
            return data1
        for feature in data2['features']:
            for feature_1 in data1:
                if feature_1['name'] == feature['name']:
                    feature_1.update(feature)
                    continue
        return data1

    @staticmethod
    def get_dataset_name_from_id(dataset_id):
        db = next(get_db())
        db_dataset = crud.get_dataset(db, dataset_id)
        return db_dataset.name

    @staticmethod
    def get_dataset_id_from_name(dataset_name,tenant_id):
        db = next(get_db())
        db_dataset = crud.get_dataset_by_name(db, dataset_name, tenant_id)
        return db_dataset.id

    @property
    def dtypes(self):
        return json.loads(self._dtypes)

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def dataset_id(self):
        return self._dataset_id

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, value):
        self._settings = value

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = value

    @property
    def relationship(self):
        return self._relationship

    @relationship.setter
    def relationship(self, value):
        self._relationship = value
    @property
    def context_variables(self):
        return self._context_variables

    @context_variables.setter
    def context_variables(self, value):
        self._context_variables = value

    @property
    def original_dataset_dim(self):
        return self._original_dataset_dim

    @property
    def original_rows(self):
        return self._original_rows

    @property
    def industry(self):
        return self._industry


    def get_details(self, detail='data',**kwargs):
        if detail == 'data':
            return self._library.read(self.dataset_name,**kwargs).data
        elif detail == 'metadata':
            return self._library.read_metadata(self.dataset_name).metadata
        elif detail == 'dtypes':
            return self._library.read_metadata(self.dataset_name).metadata.get('dtypes',{})
        elif detail == 'history':
            return self._library.read_metadata(self.dataset_name).metadata.get('history',[])
        elif detail == 'settings':
            return self._library.read_metadata(self.dataset_name).metadata.get('settings',{})
        elif detail == 'context_variables':
            return self._library.read_metadata(self.dataset_name).metadata.get('context_variables',{})
        elif detail == 'dataset_dim':
            return self._library.read(self.dataset_name,**kwargs).data.shape
        elif detail == 'rows':
            return len(self._library.read(self.dataset_name,**kwargs).data)
        elif detail == 'industry':
            return self._library.read(self.dataset_name).metadata.get('industry',None)
        elif detail == 'target':
            return self._library.read(self.dataset_name).metadata.get('target', None)
        elif detail == 'relationship':
            return self._library.read(self.dataset_name).metadata.get('relationship', None)

    def get_data(self,**kwargs):
        return self.get_details('data',**kwargs)

    def update_data(self,data,metadata=None):
        metadata = metadata if metadata else self.get_details('metadata')
        metadata['dtypes'] = json.dumps(get_column_stats(data), cls=NpEncoder)
        metadata['dataset_dim'] = data.shape
        metadata['rows'] = len(data)
        metadata['industry'] = self.get_dataset_industry_from_ai()
        self._library.write(self._dataset_name, data=data, metadata=metadata, prune_previous_versions=True)
        self._dtypes = metadata['dtypes']
        self._original_dataset_dim = metadata['dataset_dim']
        self._original_rows = metadata['rows']

    def update_target(self,target):
        metadata = self.get_details('metadata')
        data = self.get_data()
        metadata['target'] = target
        self._library.write(self._dataset_name,data=data,metadata=metadata, prune_previous_versions=True)
        self._target =target

    def update_industry(self,industry):
        metadata = self.get_details('metadata')
        data = self.get_data()
        metadata['industry'] = industry
        self._library.write(self._dataset_name,data=data,metadata=metadata, prune_previous_versions=True)
        self._industry = industry

    def update_feature_details(self):
        metadata = self.get_details('metadata')
        data = self.get_data()
        column_data = [{"name": k["name"]} for k in json.loads(self._dtypes)]
        features = get_feature_details(industry=self.industry,context=column_data)
        new_dtypes = []
        print(features.features_information_list)
        for j in features.features_information_list:
            for k in json.loads(self._dtypes):
                entry = j.dict()
                if k["name"] == entry["name"]:
                    k.update(entry)
                    new_dtypes.append(k)
        metadata['dtypes'] = json.dumps(new_dtypes, cls=NpEncoder)
        self._library.write(self._dataset_name,data=data,metadata=metadata, prune_previous_versions=True)
        self._dtypes = json.dumps(new_dtypes)

    def update_relationship(self,target,relationdata,type):
        metadata = self.get_details('metadata')
        data = self.get_data()
        metadata['relationship'] = metadata.get('relationship' ,{target: {type:None}})
        metadata['relationship'][target][type] = relationdata[target][type]
        self._library.write(self._dataset_name,data=data,metadata=metadata, prune_previous_versions=True)
        self._relationship = metadata['relationship']

    def delete_dataset(self):
        db = next(get_db())
        print(self.dataset_name)
        self._store.delete_library(self.dataset_name)
        crud.delete_dataset(db, self.dataset_id)

    def get_dtype_info( self,col:str):
        dtypes = json.loads(self._dtypes)
        return next((c for c in dtypes if c["name"] == col), None)

    def get_query(self):
        return self.get_details(detail='settings').get('query',"")

    def get_dataset_industry_from_ai(self,user_message=None):
        if user_message:
            get_industry_by_msg(user_message=user_message)
        if not self._industry:
            context = [k["name"] for k in json.loads(self._dtypes)]
            return get_industry(context)

    def get_details_from_ai(self):
        res= {}
        self.get_feature_information_from_ai('industry_prompt')
        self.get_feature_information_from_ai('categorical_prompt')
        self.get_feature_information_from_ai('duplicate_prompt')
        self.get_feature_information_from_ai('outlier_prompt')
        self.get_feature_information_from_ai('normalization_prompt')
        self.get_feature_information_from_ai('new_feature_prompt')
        self.get_feature_information_from_ai('binning_feature_prompt')
        return res

    def get_feature_information_from_ai(self,system_prompt):
        response = None
        try:
            response = json.loads(self.aiclient.run_ai_call(question=json.dumps(self._dtypes),system_prompt=system_prompt,industry=self._industry))
            self._dtypes =  json.dumps(self.update_dtypes_with_feature_information(json.loads(self._dtypes),response))
        except Exception as err:
            print(response)
            print(err)