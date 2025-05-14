from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from app import schemas, crud, models
from app.crud import create_chat_stream, get_user_by_email, get_chat_stream_by_user_id, get_chat_messages_by_stream_id, \
    create_chat_message
from app.database import get_db
from app.datasaki_ai.query import get_industry_by_msg, get_target, get_feature_details, get_missing_values, \
    get_outliers_values
from app.relationship import set_relationship_based_on_column
# from app.datasaki_ai.query import message
from app.dependencies import get_current_user
from app.arcticdb_utils import ArcticDBInstance
import app.datasaki_ai.query
import json
from app.utils import manage_missing_values_2
import logging
from app.schemas import ChatStreamSchemaCreate, ChatMessageSchema, ChatMessageSchemaCreate

logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

router = APIRouter()


@router.post("/ai/stream/{stream_id}")
async def create_stream(
        stream_id:str,
        db: Session = Depends(get_db),
        current_user: models.User = Depends(get_current_user)):
    user = get_user_by_email(db,email=current_user.username)
    chat_stream_schema = ChatStreamSchemaCreate(id=stream_id,user_id=user.id)
    create_chat_stream(db,chat_stream=chat_stream_schema)
    return response_message(stage='Initial',current_user=current_user)

@router.get("/ai/stream")
async def get_stream(
        db: Session = Depends(get_db),
        current_user: models.User = Depends(get_current_user)):
    user = get_user_by_email(db,email=current_user.username)
    return get_chat_stream_by_user_id(db,user_id=user.id)


@router.post("/ai/message/{stream_id}")
async def create_message(
        stream_id:str,
        request:dict,
        db: Session = Depends(get_db),
        current_user: models.User = Depends(get_current_user)):
    user = get_user_by_email(db,email=current_user.username)
    message = response_message(stage=request['stage'],current_user=current_user,user_message=request.get('user_message',None),dataset_id=request['dataset_id'],bypass=request.get('bypass',False),missing_data_action=request.get("missing_data_action",[]))
    chat_message_schema = ChatMessageSchemaCreate(stream_id=stream_id,user_id=user.id,message=str(message),stage=request["stage"],type=request["type"],current=True,)
    # create_chat_message(db,chat_message=chat_message_schema)
    return message

@router.get("/ai/message/{stream_id}")
async def get_message(
        db: Session = Depends(get_db),
        current_user: models.User = Depends(get_current_user)):
    user = get_user_by_email(db,email=current_user.username)
    return get_chat_messages_by_stream_id(db,stream_id=stream_id)


def response_message(stage,current_user,user_message=None,dataset_id=None,bypass=False,missing_data_action=[]):
    if stage == 'Initial':
        return [{
            "type": "text",
            "message": "Hi I am Datasaki, your Data Science Bot"
        },
            {
                "type": "text",
                "message": "To Start process please select dataset first"
            },
            {
                "type": "button",
                "message": "upload"
            },
            {
                "type": "multi_select",
                "message": "all_dataset"
            },{
            "next_stage": "data_selection"
            }]
    elif stage == "data_selection":
        data_instance = ArcticDBInstance(dataset_id=dataset_id, tenant_id=current_user.tenant_id)
        industry = data_instance.industry
        return [{
            "type": "text",
            "message": f"Your Dataset has been collected and i can see that your dataset is related to {industry} industry"
            },
            {
            "type": "text",
            "message": "If it is related to other industry please let me know i will update accordingly"
            },
            {
                "type": "text",
                "message": "If it is not correct please let me know only industry name or press continue"
            },
            {
                "type": "button",
                "message": "continue"
            },
            {
            "next_stage": "industry_update"
            }
            ]
    elif stage == "industry_update":
        if bypass:
            data_instance = ArcticDBInstance(dataset_id=dataset_id, tenant_id=current_user.tenant_id)
            industry = data_instance.industry
            column_data = [{"name":k["name"], "dtype":k["dtype"], "is_categorical":k["is_categorical"],
                            "feature_use":k["feature_use"]}
                           for k in json.loads(data_instance._dtypes)]
            return [{
            "type": "text",
            "message": f"Thank you for the information, i have updated your industry to {industry}"
            },
                {
                    "type": "text",
                    "message": "Here is the details of your dataset"
                },
                {
                    "type": "shape",
                    "message": {data_instance.original_dataset_dim}
                },
                {
                    "type": "details",
                    "dataset": column_data
                },
                {
                    "type": "text",
                    "message": "Now could you let me know what you want predict from your dataset so we can select the target"
                },
            {
            "next_stage": "target_set"
            }
            ]
        data_instance = ArcticDBInstance(dataset_id=dataset_id, tenant_id=current_user.tenant_id)
        industry = get_industry_by_msg(user_message=user_message)
        data_instance.update_industry(industry=industry)
        data_instance.update_feature_details()
        return [{
            "type": "text",
            "message": f"Thank you for the information, i have updated your industry to {industry}"
            },
            {
            "type": "text",
            "message": "If it is not correct please let me know only industry name or press continue"
            },
            {
                "type": "button",
                "message": "continue"
            },
            {
            "next_stage": "industry_update"
            }
            ]
    elif stage == "target_set":
        data_instance = ArcticDBInstance(dataset_id=dataset_id, tenant_id=current_user.tenant_id)
        if bypass:
            column_data = [{"name": k["name"], "dtype": k["dtype"], "is_categorical": k["is_categorical"],
                            "hasMissing": k["hasMissing"], "missingness_type": k["missingness_type"]}
                           for k in json.loads(data_instance._dtypes)]
            column_data = [value for value in column_data if value["hasMissing"] > 0]
            target = data_instance.target
            if len(column_data) > 0:
                missing_table = get_missing_values(context=column_data,industry=data_instance.industry)
                return [{
                    "type": "text",
                    "message": f"Your target has been set to {target}",
                },
                    {
                        "type": "text",
                        "message": f"Now we can proceed with feature EDA process",
                    },
                    {
                        "type": "text",
                        "message": f"First We will remove Missing Values",
                    },
                    {
                        "type": "details",
                        "missing_table":missing_table,
                    },
                    {
                        "next_stage": "feature_eda_missing"
                    }
                ]
            else:
                missing_table = "No missing or blank cell in you data"
                return [{
                    "type": "text",
                    "message": f"Your target has been set to {target}",
                },
                    {
                        "type": "text",
                        "message": f"Now we can proceed with feature EDA process",
                    },
                    {
                        "type": "text",
                        "message": f"First We will remove Missing Values",
                    },
                    {
                        "type": "text",
                        "message": f"{missing_table}",
                    },
                    {
                        "next_stage": "feature_eda_missing"
                    }
                ]
        industry = data_instance.industry
        column_data = [{"name": k["name"]} for k in json.loads(data_instance._dtypes)]
        target = get_target(context=column_data, industry=industry, user_message=user_message)
        data_instance.update_target(target=target)
        return [{
            "type": "text",
            "message": "Based on you query the suggested target is",
        },
            {
                "type": "button",
                "action": target
            },
            {
                "type": "text",
                "message": "If it is not correct please Choose from the list or press continue",
            },
            {
                "type": "select",
                "action": column_data
            },
            {
                "type": "button",
                "message": "continue"
            },
            {
                "next_stage": "target_set"
            }]
    elif stage == "feature_eda_missing":
        data_instance = ArcticDBInstance(dataset_id=dataset_id, tenant_id=current_user.tenant_id)
        data = data_instance.get_data()
        if len(missing_data_action) > 0:
            for value in missing_data_action:
                data_tmp = manage_missing_values_2(data=data,strategy=value['function'],column=value['feature'],**value["function_args"])
                data_instance.update_data(data=data_tmp)
        column_data = [{"name": k["name"],
                        "dtype": k["dtype"],
                        "skew": k.get("skew",None),
                        "kurt": k.get("kurt",None),
                        "is_categorical": k["is_categorical"],
                        "hasMissing": k["hasMissing"],
                        "missingness_type":k["missingness_type"],
                        "total_row":data_instance.original_dataset_dim[1]
                        } for k in json.loads(data_instance._dtypes)]
        outlier_res = get_outliers_values(context=column_data,industry=data_instance.industry)
        return [{
            "type": "text",
            "message": f"Your data has been cleaned and all the blank value has been updated",
        },
            {
                "type": "text",
                "message": f"Now we can proceed with next EDA process that is outlier removal",
            },
            {
                "type": "details",
                "outlier_removal_table":outlier_res
            },
            {
                "next_stage": "feature_eda_outlier_removal"
            }
        ]
    elif stage == "feature_eda_outlier_removal":
        data_instance = ArcticDBInstance(dataset_id=dataset_id, tenant_id=current_user.tenant_id)
        relationship,feature_list = set_relationship_based_on_column(data_instance,target="default")
        return [{
            "type": "text",
            "message": f"All outlier has been removed, now we get the relationship between target and feature values",
        },
            {
                "type": "text",
                "message": f"The feature that are make impact on target is :{feature_list}",
            },
            {
                "type": "text",
                "message": f"Now We are converting categorical features to dummies and it is the final dataset and there relationships",
            },
            {
                "next_stage": "feature_eda_relationship"
            }
        ]