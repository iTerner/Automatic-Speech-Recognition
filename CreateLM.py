# write this lines in the terminal
!pip -q install https://github.com/kpu/kenlm/archive/master.zip pyctcdecode
! sudo apt -y install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
! wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
! mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2
! ls kenlm/build/bin

with open("text.txt", 'w') as f:
    with open("train_transcription.txt", "r") as g:
        for i, sen in enumerate(g.readlines()):
              # print(sen)
              splited_word = []
              for letter in sen:
                  # print(letter)
                  if not letter in ["\n"]:
                    splited_word.append(letter)
                    splited_word.append(" ")
              # splited_word.append("\n")
              splited_word[-1] = "\n"
              f.writelines(splited_word)

# write this line in the terminal - make sure the path is correct!
!kenlm/build/bin/lmplz --order 3 -S 2G -T /content/ --discount_fallback  < /content/text.txt > lang_model.arpa