import re
import matplotlib.pyplot as plt
def extractor(tag):
    
    matches = re.match(r'(\d+):(\d+):(.+)', tag)
    
    if matches:
        try : 
            number1 = matches.group(1)
            number2 = matches.group(2)
            text = matches.group(3)
        except : 
            number1 = None
            number2 = None
            text = None
    return number1, number2, text

def preprocessing(tags):
    all_tags = tags.split(',')
    tags = []
    for tag in all_tags:
        if tag != '':
            tags.append(tag)
            
    res = []
    for tag in tags:        
        n1, n2, txt = extractor(tag)
        res.append([n1, n2, str(txt)])
    return res

def clean_text(input_text):
    alphanumeric_text = re.sub(r'[^a-zA-Z0-9\s]', '', input_text)

    clean_text = alphanumeric_text.strip()

    return clean_text

def generate_tags_1(data):
    targ_dict = []
    for i in range(len(data['tags_cleaned'])):
        tags = data.iloc[i]['tags_cleaned']
        text = data.iloc[i]['text']
        temp_dict = {}
        for tag in tags:
            start = tag[0]
            end = tag[1]
            targ_tag = tag[2]
            targ_word = text[int(start)-1:int(end)]            
            targ_word = clean_text(targ_word)
            targ_words = targ_word.split(' ')
            for word in targ_words:
                temp_dict[word] = targ_tag
        targ_dict.append(temp_dict)
    return targ_dict

def generate_tags_2(data):
    all_res = []
    for i in range(len(data['mapping'])):
        texts = [clean_text(w) for w in (data.iloc[i]['text'].split(' '))]
        mappings = data.iloc[i]['mapping']
        result = []
        for word in texts:
            if word in list(mappings.keys()):
                result.append((word, mappings[word]))
            else :
                result.append((word, 'o'))
        all_res.append(result)
    return all_res

def create_dataset(g1):
    g1['mapping'] = generate_tags_1(g1)
    g1['all_mapping'] = generate_tags_2(g1)
    
    return g1

def get_plots(history):
    save_path = r"D:\ML-Projects\Continual-Learning\training plots"
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
# Create subplots
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
