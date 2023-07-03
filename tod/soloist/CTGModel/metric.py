
def isPolluted(sentence):
    is_fuck=0
    if "fuck" in sentence or "Fuck" in sentence or "fucking" in sentence:
        is_fuck=1
        return 1

    is_offensive_prefix=0
    offensive_list=["BS","crap","that mouth","dare","hell","devil","bitch",
                    "son of","Son of","Damn","big mouth","beast","sick","hate",
                    "bother"]
    for word in offensive_list:
        if word in sentence:
            is_offensive_prefix=1
            return 1

    is_offensive_role=0
    dirty_role_ls=["nigger","negro","chink","spic","honky","kraut","kike",
                   "bitch","cunt","faggot","dyke"]
    for dr in dirty_role_ls:
        if dr in sentence:
            is_offensive_role=1
            return 1

    return 0.0

















