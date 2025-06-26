
import json
# wiKi = open("wikipedia-en-sentences.txt", "r")

# tokenisedSenteces = open("wikiSentences.txt", "w")

# for line in wiKi:
#     tokenisedSenteces.write("<BOS> {} <EOS>\n".format(line[:-1]))

# wiKi.close()
# tokenisedSenteces.close()



redit = open("reddit_title_text_2020.jsonl", "r")
# FineTuneSentences = open("redditSentences.txt", "w")
for line in redit:
    json_line = json.loads(line)
    print("<System>" + json_line.get("title") + "<EOS>")
    print("<User>" +json_line.get("body") + "<EOS>")

    break

#  do something like this when training 
# trainX = "<System>" + json_line.get("title") + "<EOS>"
# trainY = "<User>" + json_line.get("body") + "<EOS>"

redit.close()