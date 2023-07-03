import re

def getSoloistSlots():
    """This function is used to ..."""
    result_set=[]
    fraction_ls=[0.01,0.02,0.04,0.06,0.08,0.1]
    num_ls=[1,2,3,4,5]
    for fraction in fraction_ls:
        for num in num_ls:
            filepath=f"../{fraction}_soloist_predict_files_{num}.txt"
            temset=parserSlots(filepath)
            result_set.extend(list(temset))

    result_set=set(result_set)
    print(result_set)
    """
    running results
{'[food]', '[address]', '[price]', '[fee]', '[rating]', '[car]', '[time]', '[stars]', '[type]', '[location]', '[id]', '[to]', '[duration]', '[phone]', '[postcode]', '[fare]', '[area]', '[reference]', '[destination]', '[from]', '[amenity]', '[date]', '[name]', '[stay]', '[city]', '[departure]'}
    """

    return 0


def getDamdSlots():
    """This function is used to ..."""
    result_set=[]
    fraction_ls=[0.01,0.02,0.04,0.06,0.08,0.1]
    num_ls=[1,2,3,4,5]
    for fraction in fraction_ls:
        for num in num_ls:
            filepath=f"../damd/{fraction}_sclstm_predict_files_{num}.txt"
            temset=parserSlots(filepath)
            result_set.extend(list(temset))

    result_set=set(result_set)
    print(result_set)
"""
{'[value_choice]', '[value_pricerange]', '[value_postcode]', '[value_departure]', '[value_address]', '[value_destination]', '[hotel]', '[value_car]', '[train]', '[value_phone]', '[attraction]', '[value_time]', '[value_name]', '[value_stay]', '[restaurant]', '[value_leave]', '[value_people]', '[value_food]', '[taxi]', '[value_area]', '[value_day]', '[value_reference]', '[value_arrive]', '[value_price]', '[value_id]', '[value_stars]', '[value_type]'}
"""
    return 0


def parserSlots(filename):

    pattern1=re.compile(r"\[.*?\]")

    slotls=[]
    with open(filename,'r',encoding="utf8") as f:
        data=f.readlines()
    for line in data:
        candidates=re.findall(pattern1,line)
        slotls.extend(candidates)

    return set(slotls)

if __name__=="__main__":
    # getSoloistSlots()
    getDamdSlots()
