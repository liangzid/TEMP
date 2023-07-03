import math




inp,labels,positive,belief_end,context_end,database_end=backInferenceFormat(inp,
                                                                                        labels,
                                                                                        positive,
                                                                                        belief_end,
                                                                                        context_end,
                                                                                        database_end)


def backInferenceFormat(inp,labels,positive,belief_end,context_end,database_end):
    """This function is used to rerange the back-inference sequence."""
    
    
    pass
    return inp,labels,positive,belief_end,context_end,database_end



def sentence2BackSentenceWindow(sentence,max_seq_length,
                                window_length,target_length,slide_step,
                                is_paddle,sep_token=1,pad_token=0):
    """generate a list of back sentence prediction requirements.

    sentence: input ids list.
    msl: no use
    window_length: prefix length
    target_length: generated sentence length
    slide_step: slide step
    is_paddle: is pad or not, boolean
    sep_token: septoken
    pad_token: padtoken
    
    """

    sent_len=len(sentence)
    # L_pad + sent_len = window_length + N* slide_step
    # N= ceil((sent_len - window_length)/slide_step)
    # L_pad= N*slide_step - sent_len + window_length
    ## num sequence
    N= math.ceil((sent_len - window_length-target_length)/slide_step)
    num_pad= N*slide_step - sent_len + window_length + target_length

    new_sentence=[pad_token for _ in range(num_pad)]+sentence
    new_sent_len=len(new_sentence)
    sent_len=new_sent_len
    
    back_sent_list=[]
    
    for i in range(N):
        window_begin=sent_len - i*slide_step-window_length
        window_end=sent_len - i*slide_step # not tourched

        target_begin=window_begin-target_length
        target_end=window_begin
        back_sent_list.append(sentence[window_begin:window_end]+[sep_token]+sentence[target_begin:target_end])
            
    return back_sent_list


def sentence2BackSentenceAccumulated(sentence,max_seq_length,
                                window_length,target_length,slide_step,
                                     is_paddle,sep_token=1,pad_token=0):
    """generate a list of back sentence prediction requirements."""

    sent_len=len(sentence)
    # L_pad + sent_len = window_length + N* slide_step
    # N= ceil((sent_len - window_length)/slide_step)
    # L_pad= N*slide_step - sent_len + window_length
    ## num sequence
    N= math.ceil((sent_len - window_length-target_length)/slide_step)
    num_pad= N*slide_step - sent_len + window_length + target_length
    new_sentence=[pad_token for _ in range(num_pad)]+sentence
    new_sent_len=len(new_sentence)
    sent_len=new_sent_len
    
    back_sent_list=[]
    
    for i in range(N):
        window_begin=sent_len - i*slide_step-window_length
        window_end=sent_len # not tourched

        target_begin=window_begin-target_length
        target_end=window_begin
        back_sent_list.append(sentence[window_begin:window_end]+[sep_token]+sentence[target_begin:target_end])
            
    return back_sent_list












