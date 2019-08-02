# seq2seq로 구현한 한국어 일상대화 챗봇 모델

### - Preprocessing

- tokenize: 단어(띄어쓰기) 단위
- text normalize
  - `'?', '!', '.'` 를 제외한 특수기호 제거
- `SOS`, `EOS`, `UNK`, `PAD` 특수토큰 사용
  - 모르는 단어가 들어왔을 시 `UNK` 로 처리
  - 가장 긴 길이의 발화보다 짧은 발화는 `PAD` 로 zeropadding 처리



### - Model Architecture

- embedding

- - `torch.nn.Embedding` layer 사용 
  - 임베딩 차원: 학습한 데이터에 있는 단어 개수 (8700개)

- encoder

  - `torch.nn.GRU` layer 사용
  - hidden_layer = 2
  - dropout = 0.5

- decoder

  - attention layer 사용
  - hidden_layer = 2
  - dropout = 0.5
  - softmax 값으로 출력

- loss function

  - `maskNLL`(Negative Log Likelihood) Loss 사용 - batch별로 다른 길이의 input 처리

- training

  - SGD(mini-batch)
    - batch_size = 64
  - gradient clipping
  - random teacher forcing 
    - teacher forcing 하지 않을 경우 greedy decoding
  - train_iteration = 4000

- max_length = 20

  - 출력 발화 길이 제한

  

### - Requirements

```shell
  1 pandas
  2 re
  3 torch
  4 random
  5 os
  6 sys
  7 tqdm
```



# Testing Chatbot

- 실행

  ```shell
  $ python3 model.py
  > [input]
  ```

- 종료

  `quit` 혹은 `q` 입력 시 종료

  ```shell
  $ python3 model.py
  > [input]
  Bot: [output]
  > quit
  ```

  

- 실행 예시

![image](https://user-images.githubusercontent.com/44221498/62372203-cdc9d800-b571-11e9-9b23-16e1abfbc0fd.png)







​	