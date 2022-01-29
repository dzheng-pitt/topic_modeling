import numpy as np
import os
import re
from nltk.corpus import stopwords
import time
import scipy.special as sc
import scipy as scipy
import matplotlib.pyplot as plt


def parse_clean_data(path, doc_len = 125):
    
    # takes a folder of presidential addresses and cleans the data into
    # 125 (default) word documents. 125 words is about a paragraph to convey
    # one idea
    
    # get stopwords and data files
    stopwords_dict = stopwords.words('english')
    files = os.listdir(path)
    
    # init data and preallocate some memory
    word_to_num_dict = {}
    num_to_word_dict = {}
    len_vocab = 0
    data_tuple = []
    doc_count = 0
    documents = np.zeros((10000, doc_len))
    t = np.zeros(10000)
    
    # for each file
    for i in range(len(files)):
        
        # record the president and year
        president_year = files[i].split('_')
        president = president_year[0]
        year = president_year[1].split('.')[0]
        
        # open the address
        filename = os.path.join(path, files[i])
        text_file = open(filename, "r").read().replace('\n', ' ')
        word_vec = np.empty(doc_len)
        word_num = 0
        
        # for each word in the address
        for word in text_file.split(' '):
            
            # clean it and see if it is a stop word
            clean_word = re.sub('[^A-Za-z0-9]+', '', word).lower() 
            if clean_word in stopwords_dict or clean_word == '':
                continue
            
            #  add to dictionary if not in currently in
            elif clean_word not in word_to_num_dict:
                
                word_to_num_dict[clean_word] = len_vocab
                num_to_word_dict[len_vocab] = clean_word
                len_vocab += 1
            
            # add to document
            word_vec[word_num] = word_to_num_dict[clean_word]
            word_num += 1
            
            # document word length reached so we record this data point
            if word_num == doc_len:
                
                data_tuple.append((president, year, doc_count, word_vec))
                t[doc_count] = year
                documents[doc_count,:] = word_vec
                word_vec = np.empty(doc_len)
                word_num = 0
                doc_count += 1   
                
            # anything remaining from an address is not included
    
    # clean up documents
    documents = documents[0:doc_count,:].astype('int')
    
    return documents, t, data_tuple, num_to_word_dict

def write_most_common_words(topics, beta, num_to_word_dict):
    
    # for each topic, find and write the top 20 words that occur based on beta
    string = ''
    for topic in range(topics):
        string = string + 'Topic:' + str(topic) + '\n'
        topic_words = beta[topic,:].argsort()[-20:]
        for idx in reversed(topic_words):
            string = string + num_to_word_dict[idx] + '\n'
        string = string + '\n'
        
    text_file = open("output/most_common_words_by_topic.txt", "w")
    text_file.write(string)
    text_file.close()
    
def plot_topics_over_time(t, min_t, max_t, topics, gamma, psi):
    
    years = t*(max_t-min_t)+min_t
    year_topic_density = np.zeros((len(np.unique(years)),topics))

    # calculate the empricial topic density by year given gamma
    for i in range(len(np.unique(years))):
        for topic in range(topics):
            year = np.sort(np.unique(years))[i]
            year_topic_density[i,topic] = np.sum(gamma[np.where(years==int(year)),topic])
    
    year_topic_density = year_topic_density/np.max(year_topic_density,axis=0).reshape(1,50)
    x = np.array([i for i in range((max_t.astype('int')- min_t.astype('int') - 2))])/(max_t.astype('int')-min_t.astype('int')-3)
    
    # plot the beta distribution according to psi and the empricial topic density
    for topic in range(topics):
         
        ys = scipy.stats.beta(psi[topic,0],psi[topic,1]).pdf(x)
        max_ys = np.max(ys[ys<10000])
        ys = ys/max_ys
        plt.plot(x*(max_t-min_t)+min_t,ys)
        plt.bar(x*(max_t-min_t)+min_t,year_topic_density[:,topic])
        plt.title(topic)
        plt.savefig('output/topic_'+str(topic)+'.png')
        plt.show()
            
if __name__ == '__main__':

    np.random.seed(10)
    
    documents, t, data_tuple, num_to_word_dict = parse_clean_data('data')

    # params and init model
    
    # corpus size
    topics = 50
    num_docs = documents.shape[0]
    doc_len = documents.shape[1]
    vocab_len = len(num_to_word_dict)
    
    # time index from 0 to 1 for beta distribution
    t = t[0:num_docs]
    min_t = min(t) - 1
    max_t = max(t) + 1
    t = (t - min_t)/(max_t-min_t)
    
    # model parameters
    alpha = np.ones((1,topics))*0.1
    beta = np.random.uniform(size =(topics, vocab_len))
    beta_n = np.sum(beta, axis = 1).reshape(topics,1)
    beta = beta/beta_n
    psi = np.random.uniform(size=(topics,2))

    phi = np.ones((num_docs, doc_len, topics))*(1/topics)
    gamma = np.repeat(alpha + doc_len/topics, num_docs, axis = 0)
    
    # loop
    max_iters = 100
    iters = 0
    time_info = time.time()
    
    # fit with EM
    while True:
    
        # E step
        E_iters = 0
        while True:
            old_phi = phi.copy()
            old_gamma = gamma.copy()
                                                                                   
            for doc in range(num_docs):
                exp_E_log_theta = np.exp(sc.digamma(old_gamma[doc, :]))   	
                exp_C = t[doc]**psi[:,0]*(1-t[doc])**(psi[:,1]-1)/sc.beta(psi[:,0],psi[:,1])
                phi[doc, :,:] = exp_E_log_theta*exp_C*beta[:, documents[doc,:]].transpose()
            phi = phi/np.repeat(np.sum(phi,axis=2).reshape((num_docs,doc_len,1)),topics,axis=2)
            gamma = alpha + np.sum(phi, axis = 1)        
            
            criteria_e = (1/(2*num_docs))*(np.linalg.norm(phi-old_phi) + \
                                           np.linalg.norm( gamma - old_gamma))
            E_iters += 1
            if criteria_e <= 0.01:
                break
        
        # beta M step
        beta = np.zeros((topics,vocab_len))
        for doc in range(num_docs):
            for n in range(doc_len):
                vcol = documents[doc,n]
                beta[:,vcol] += phi[doc,n,:]
        beta_n = np.sum(beta, axis = 1).reshape(topics,1)
        beta = beta/beta_n
        
        # alpha step
        alpha_iters = 0
        while True:
            old_alpha = alpha.copy()
        
            g1 = sc.digamma(np.sum(alpha)) - sc.digamma(alpha)
            g2 = np.sum(sc.digamma(gamma) - \
                        sc.digamma(np.sum(gamma,axis=1)).reshape(num_docs,1),axis=0)
            g = num_docs*g1 + g2
            h = num_docs*sc.polygamma(1,alpha)
            z = -sc.polygamma(1,np.sum(alpha))
            c = np.sum(g/h)/(z**-1 + np.sum(h**-1))   
            alpha = alpha + (g - c)/h
                
            criteria_alpha = np.linalg.norm(alpha-old_alpha)
            alpha_iters +=1
            if criteria_alpha <= 0.0001 :
                break
        
        # psi step
        psi_iters = 0
        while True:
            old_psi = psi.copy()
                     
            for topic in range(topics):
                t_repeat = np.repeat(np.log(t).reshape(num_docs, 1), doc_len, axis = 1)
                t_repeat_1 = np.repeat(np.log(1-t).reshape(num_docs, 1), doc_len, axis = 1) 
                
                g_psi_1 = np.sum(np.sum(phi[:, :, topic]*t_repeat, axis =1)) -\
                    np.sum(np.sum(phi[:,:,topic]*(sc.digamma(psi[topic, 0])-sc.digamma(np.sum(psi[topic,:]) ))))
                
                g_psi_2 = np.sum(np.sum(phi[:, :, topic]*t_repeat_1, axis =1)) -\
                    np.sum(np.sum(phi[:,:,topic]*(sc.digamma(psi[topic, 1])-sc.digamma(np.sum(psi[topic,:]) ))))
                g_psi= np.array([g_psi_1 , g_psi_2])
                
                h_psi_1 = np.sum(np.sum(phi[:,:, topic] *sc.polygamma(1, psi[topic, 0]))) 
                h_psi_2 = np.sum(np.sum(phi[:,:, topic] *sc.polygamma(1, psi[topic, 1])))   
                
                h_psi = np.array([h_psi_1, h_psi_2])
            
                z_psi = -np.sum(np.sum(phi[:,:, topic] *sc.polygamma(1, psi[topic, 0] + psi[topic, 1]))) 
                
                c_psi = np.sum(g_psi/h_psi)/( z**-1 + np.sum(h_psi**-1))
                psi[topic, :] = psi[topic, :] + (g_psi-c_psi)/h_psi
                
                if psi[topic, 0] < 1e-8:
                    psi[topic, 0] = 1e-8
                
                if psi[topic, 1] < 1e-8:
                    psi[topic, 1] = 1e-8
                
                if psi[topic, 0] > 10000:
                    psi[topic, 0] = 10000
                
                if psi[topic, 1] > 10000:
                    psi[topic, 1] = 10000         
            
            criteria_psi = np.linalg.norm(psi-old_psi)
            psi_iters +=1
            if criteria_psi <= 0.001 or psi_iters == max_iters:
                break
        
        
        # some info each iteration
        iters += 1
            
        if iters % 1 == 0:
            end_time = round(time.time() - time_info)
            print('overall iters:',iters, 
                  'seconds:',end_time,
                  'expectation iters:',E_iters,
                  'alpha iters:',alpha_iters,
                  'psi iters:',psi_iters)
            time_info = time.time()
                
        if (E_iters <= 1 and alpha_iters <= 1 and psi_iters <= 1) or iters == max_iters:
            break
    
    # review output   
    write_most_common_words(topics, beta, num_to_word_dict)
    plot_topics_over_time(t, min_t, max_t, topics, gamma, psi)
    
    