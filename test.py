from dotenv import dotenv_values
import os
import time
from utils import get_embeddings, dim_reduc,clustering, vocab_builder, find_describer
import numpy as np

config = {
    **dotenv_values(".env.local"),
    **os.environ  # load shared development variables
}

corpus = """The AIM-9 Sidewinder (where "AIM" stands for "Air Intercept Missile") is a short-range air-to-air missile which entered service with the United States Navy in 1956, and subsequently was adopted by the US Air Force in 1964. Since then, the Sidewinder has proved to be an enduring international success, and its latest variants remain standard equipment in most Western-aligned air forces.[3] The Soviet K-13 (AA-2 'Atoll'), a reverse-engineered copy of the AIM-9B, was also widely adopted by a number of nations.

Low-level development started in the late 1940s, emerging in the early 1950s as a guidance system for the modular Zuni rocket.[4][5] This modularity allowed for the introduction of newer seekers and rocket motors, including the AIM-9C variant, which used semi-active radar homing and served as the basis of the AGM-122 Sidearm anti-radar missile. Originally a tail-chasing system, early models saw extensive use during the Vietnam War but had a low success rate. This led to all-aspect capabilities in the L version which proved to be an extremely effective weapon during combat in the Falklands War and the Operation Mole Cricket 19 ("Bekaa Valley Turkey Shoot") in Lebanon. Its adaptability has kept it in service over newer designs like the AIM-95 Agile and SRAAM that were intended to replace it.

The Sidewinder is the most widely used air-to-air missile in the West, with more than 110,000 missiles produced for the U.S. and 27 other nations, of which perhaps one percent have been used in combat. It has been built under license by some other nations including Sweden, and can even equip helicopters, such as the Bell AH-1Z Viper. The AIM-9 is one of the oldest, lowest cost, and most successful air-to-air missiles, with an estimated 270 aircraft kills in its history of use.[6]

The United States Navy hosted a 50th-anniversary celebration for the Sidewinder in 2002. Boeing won a contract in March 2010 to support Sidewinder operations through to 2055, guaranteeing that the weapons system will remain in operation until at least that date. Air Force Spokeswoman Stephanie Powell noted that due to its relatively low cost, versatility, and reliability it is "very possible that the Sidewinder will remain in Air Force inventories through the late 21st century"""
corpus = corpus.split(".")
# corpus = ["Hello my name is John", "Hey how are you", "Bye I am leaving", "Good meeting you but I must go", "Hello what is this?", "I have to leave now", "I am heading out"]


done = False
j = 0
while j < 5 and not done:
    embeddings = get_embeddings(corpus, config["HF_KEY"])
    print(embeddings)
    j+=1
    # print(type(embeddings) is np.ndarray)
    done = type(embeddings) == np.ndarray
    
    time.sleep(1)


dos_dims = dim_reduc(embeddings)
clustered = clustering(dos_dims)
out = []
for label in np.unique(clustered):
    sentences = ".".join(np.array(corpus)[clustered == label])
    # sec_embed = get_embeddings(sentences, config["HF_KEY"])
    print(find_describer(sentences))



# vocab = vocab_builder(corpus)

# find_describer(embeddings)
