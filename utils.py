def get_available_openai_models(put_first=None, filter_by=None):
    model_list = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview']
    if filter_by:
        model_list = [m for m in model_list if filter_by in m]
    if put_first and put_first in model_list:
        model_list.remove(put_first)
        model_list = [put_first] + model_list
    return model_list
