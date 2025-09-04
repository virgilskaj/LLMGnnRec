
import threading
import openai
import time
import pandas as pd
import pickle
import os
import numpy as np
import torch

# openai.api_key = ""
openai.api_key = ""

import requests

file_path = ""


# # MovieLens
# def construct_prompting(item_attribute, indices): 
#     # pre string
#     pre_string = "You are now a search engines, and required to provide the inquired information of the given movies bellow:\n"
#     # make item list
#     item_list_string = ""
#     for index in indices:
#         title = item_attribute['title'][index]
#         genre = item_attribute['genre'][index]
#         item_list_string += "["
#         item_list_string += str(index)
#         item_list_string += "] "
#         item_list_string += title + ", "
#         item_list_string += genre + "\n"
#     # output format
#     output_format = "The inquired information is : director, country, language.\nAnd please output them in form of: \ndirector::country::language\nplease output only the content in the form above, i.e., director::country::language\n, but no other thing else, no reasoning, no index.\n\n"
#     # make prompt
#     prompt = pre_string + item_list_string + output_format
#     return prompt 

# Netflix
def construct_prompting(item_attribute, indices): 
    # pre string
    pre_string = "You are now a search engines, and required to provide the inquired information of the given movies bellow:\n"
    # make item list
    item_list_string = ""
    for index in indices:
        year = item_attribute['year'][index]
        title = item_attribute['title'][index]
        item_list_string += "["
        item_list_string += str(index)
        item_list_string += "] "
        item_list_string += str(year) + ", "
        item_list_string += title + "\n"
    # output format
    output_format = "The inquired information is : director, country, language.\nAnd please output them in form of: \ndirector::country::language\nplease output only the content in the form above, i.e., director::country::language\n, but no other thing else, no reasoning, no index.\n\n"
    # make prompt
    prompt = pre_string + item_list_string + output_format
    return prompt

### chatgpt attribute generate
def LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt):
    if indices[0] in augmented_attribute_dict:
        return 0
    else:
        try: 
            print(f"{indices}")
            prompt = construct_prompting(toy_item_attribute, indices)
            url = "https://api.openai.com/v1/completions"
            headers={
                # "Content-Type": "application/json",
                "Authorization": "Bearer your key"
            }

            params={
                "model": "text-davinci-003",
                "prompt": prompt,
                "max_tokens": 1024,
                "temperature": 0.6,
                "stream": False,
            } 

            response = requests.post(url=url, headers=headers,json=params)
            message = response.json()

            content = message['choices'][0]['text']
            print(f"content: {content}, model_type: {model_type}")

            rows = content.strip().split("\n")  # Split the content into rows
            for i,row in enumerate(rows):
                elements = row.split("::")  # Split each row into elements using "::" as the delimiter
                director = elements[0]
                country = elements[1]
                language = elements[2]
                augmented_attribute_dict[indices[i]] = {}
                augmented_attribute_dict[indices[i]][0] = director
                augmented_attribute_dict[indices[i]][1] = country
                augmented_attribute_dict[indices[i]][2] = language
            # pickle.dump(augmented_sample_dict, open('augmented_sample_dict','wb'))
            pickle.dump(augmented_attribute_dict, open(file_path + 'augmented_attribute_dict','wb'))
        
        # except ValueError as e:
        except requests.exceptions.RequestException as e:
            print("An HTTP error occurred:", str(e))
            error_cnt += 1
            if error_cnt==5:
                return 1
            LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
        except ValueError as ve:
            print("ValueError error occurred while parsing the response:", str(ve))
            # time.sleep(25)
            error_cnt += 1
            if error_cnt==5:
                return 1
            # print(content)
            LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
        except KeyError as ke:
            print("KeyError error occurred while accessing the response:", str(ke))
            # time.sleep(25)
            error_cnt += 1
            if error_cnt==5:
                return 1
            # print(content)
            LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
        except IndexError as ke:
            print("IndexError error occurred while accessing the response:", str(ke))
            return 1
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            # time.sleep(25)
            error_cnt += 1
            if error_cnt==5:
                return 1
            # print(content)
            LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
        return 1







### chatgpt attribute embedding
def LLM_request(toy_augmented_item_attribute, indices, model_type, augmented_atttribute_embedding_dict, error_cnt):
    for value in augmented_atttribute_embedding_dict.keys():
        print(value)
        if indices[0] in augmented_atttribute_embedding_dict[value]:
            # return 0
            continue 
        else:
            try: 
                print(f"{indices}")
                # prompt = construct_prompting(toy_item_attribute, indices)
                url = "https://api.openai.com/v1/embeddings"
                headers={
                    # "Content-Type": "application/json",
                    "Authorization": "Bearer your key"
                }
                params={
                "model": "text-embedding-ada-002",
                "input": toy_augmented_item_attribute[value][indices].values[0]
                }

                response = requests.post(url=url, headers=headers,json=params)
                message = response.json()

                content = message['data'][0]['embedding']

                augmented_atttribute_embedding_dict[value][indices[0]] = content
                # pickle.dump(augmented_sample_dict, open('augmented_sample_dict','wb'))
                pickle.dump(augmented_atttribute_embedding_dict, open(file_path + 'augmented_atttribute_embedding_dict','wb'))
            
            # except ValueError as e:
            except requests.exceptions.RequestException as e:
                print("An HTTP error occurred:", str(e))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt)
            except ValueError as ve:
                print("An error occurred while parsing the response:", str(ve))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt)
            except KeyError as ke:
                print("An error occurred while accessing the response:", str(ke))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt)
            except Exception as ex:
                print("An unknown error occurred:", str(ex))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt)
            # return 1










def file_reading():
    augmented_atttribute_embedding_dict = pickle.load(open(file_path + 'augmented_atttribute_embedding_dict','rb')) 
    return augmented_atttribute_embedding_dict

### baidu attribute embedding
def LLM_request(toy_augmented_item_attribute, indices, model_type, augmented_atttribute_embedding_dict, error_cnt, key, file_name):
    for value in augmented_atttribute_embedding_dict.keys():
        if indices[0] in augmented_atttribute_embedding_dict[value]:
            # return 0
            continue
        else:
            try: 
                print(f"{indices}")
                print(value)

                ### chatgpt #############################################################################################################################
                # prompt = construct_prompting(toy_item_attribute, indices)
                url = "https://api.openai.com/v1/embeddings"
                headers={
                    # "Content-Type": "application/json",
                    "Authorization": "Bearer your key"
                }
                ### chatgpt #############################################################################################################################


                params={
                "model": "text-embedding-ada-002",
                "input": str(toy_augmented_item_attribute[value][indices].values[0])
                }
                response = requests.post(url=url, headers=headers,json=params)
                message = response.json()

                content = message['data'][0]['embedding']

                augmented_atttribute_embedding_dict[value][indices[0]] = content
                pickle.dump(augmented_atttribute_embedding_dict, open(file_path + file_name,'wb'))
            
            # except ValueError as e:
            except requests.exceptions.RequestException as e:
                print("An HTTP error occurred:", str(e))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt, key, file_name)
            except ValueError as ve:
                print("An error occurred while parsing the response:", str(ve))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt, key, file_name)
            except KeyError as ke:
                print("An error occurred while accessing the response:", str(ke))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt, key, file_name)
            except Exception as ex:
                print("An unknown error occurred:", str(ex))
                time.sleep(5)
                # print(content)





