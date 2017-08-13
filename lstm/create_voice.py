from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import requests, threading, wave, os.path, os, sys, Queue, time

SERVER_ADDRESS = "http://localhost"
SERVER_PORT = [59125, 59126, 59127, 59128]

OUTPUT_FILE = "output/output.wav"
OUTPUT_SENTENCE_FILE_PREFIX = 'output/output_'

EMOTIONS = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']      # value
emotions_from_data = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral']  # key
emotion_mapper = {emotions_from_data[i]:EMOTIONS[i] for i in range(len(EMOTIONS))}

def get_param_input_text(sentence, emotion):
    emotion = emotion_mapper[emotion]
    if emotion not in EMOTIONS:
        print "FATAL ERROR! cannot synthesize voice for this emotion"
        print "received:", emotion
        print "expected:", "/".join(EMOTIONS)
        sys.exit()

    node_emotionml = Element('emotionml')
    node_emotionml.set("version", "1.0")
    node_emotionml.set("xmlns", "http://www.w3.org/2009/10/emotionml")
    node_emotionml.set("category-set", "http://www.w3.org/TR/emotion-voc/xml#big6")

    if emotion != 'neutral':

        node_emotion = SubElement(node_emotionml, 'emotion')
        node_emotion.text = sentence

        node_category = SubElement(node_emotion, 'category')
        node_category.set("name", emotion)
    else:
        node_emotionml.text = sentence

    return str(tostring(node_emotionml))

def get_payload(sentence, emotion):
    payload = {
    "INPUT_TYPE":"EMOTIONML", 
    "OUTPUT_TYPE": "AUDIO",
    "AUDIO":"WAVE_FILE",
    "LOCALE":"en_US"
    }

    param_input_text = get_param_input_text(sentence, emotion)

    payload['INPUT_TEXT'] = param_input_text
    '''
    input_type will be now emotionml
    and input_text will be xml representation
    '''
    return payload

def get_server_url(index):
    server_port = str(SERVER_PORT[index%(len(SERVER_PORT))])
    server_url = SERVER_ADDRESS+":"+server_port+"/process"
    return server_url

def get_wav_from_server(payload, index):
    response = requests.get(get_server_url(index), params = payload, timeout=15)
    status_code = response.status_code
    print "status_code = "+str(status_code)
    if response.status_code != 200:
        print "BAD REQUEST given by url "+repr(response.url)
        print response.text
        return 0
    content = response.content # this is in binary format
    return content
    
def get_sentence_file_name(index):
    padder = 10000
    file_suffix = padder + index;
    file_suffix = str(file_suffix)[1:]
    file_name = OUTPUT_SENTENCE_FILE_PREFIX+file_suffix+".wav"
    return file_name

def write_wav_file(wav_object, sentence_file_name):
    wav_file = open(sentence_file_name, 'wb')
    wav_file.write(wav_object)
    wav_file.close()

def get_wav_file(sentence, emotion, sentence_files, index):
    sentence_file_name = get_sentence_file_name(index)
    # if os.path.isfile(sentence_file_name):
    #     sentence_files[index] = sentence_file_name        
    #     return

    payload = get_payload(sentence, emotion)
    print "Thread "+str(index)+" fetched the payload"

    wav_object = get_wav_from_server(payload, index)
    print "Thread "+str(index)+" fetched the wav_object of length "+str(len(wav_object))

    
    sentence_files[index] = sentence_file_name
    write_wav_file(wav_object, sentence_file_name)
    #print "Thread",index,":\nwav_object size:",len(sentence_files)

def start_thread(sentence, emotion, sentence_files, index):
    thread = threading.Thread(target = get_wav_file, args = (sentence, emotion, sentence_files, index))
    return thread

def start_threads(sentences, emotions, sentence_files):
    thread_list = Queue.Queue(maxsize=0)
    for index in range(len(sentences)):
        if index>=4:
            previous_thread = thread_list.get()
            previous_thread.join()
        thread = start_thread(sentences[index], emotions[index], sentence_files, index)
        print 'now starting thread', index
        thread.start()
        thread_list.put(thread)
        #thread.join()
    return thread_list

def wait_for_all_threads_to_join(thread_list):
    print 'now waiting for threads to join'
    for index in range(len(thread_list))    :
        thread = thread_list[index]
        print 'now waiting for thread',index
        thread.join()

def construct_final_wav_file(sentence_files):
    data= []
    for file_name in sentence_files:
        # print 'filename:',file_name
        wav_file = wave.open(file_name, 'rb')
        data.append( [wav_file.getparams(), wav_file.readframes(wav_file.getnframes())] )
        wav_file.close()

    output = wave.open(OUTPUT_FILE, 'wb')
    #print "the data is:", data
    output.setparams(data[0][0])
    for i in range(len(sentence_files)):
        output.writeframes(data[i][1])
    output.close()

def start_audio_file_generation(sentences, emotions):
    sentence_files = [0 for _ in sentences]
    thread_list = start_threads(sentences, emotions, sentence_files)
    #wait_for_all_threads_to_join(thread_list)
    #All files are not generated at the rate of
    #their consumption by the construct_final_wav_file()
    time.sleep(2)
    construct_final_wav_file(sentence_files)
    # now delete the individual sentence files here
    for sentence_file_name in sentence_files:
       os.remove(sentence_file_name)
print get_param_input_text("hi hello are you","joy")
print get_wav_from_server(get_payload("hi hello are you","joy"),1)
