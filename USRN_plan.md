# USRN_Plan

## Experiment

### Application

- [ ] Variance map interpolation
  - [ ] Log_var -> exp -> sigmoid 

- [ ] classification 하고 연결 시켜보기

  - [ ] sr 된 영상을 classification에 넣고 accuracy 변화 확인

  - [ ] Variance map을 사용하고 안하고 차이 확인

    - [ ] 이거에서 유의미한 의미 찾으면 ㄱ ㄱ

    

### Uncertainty Meaning

- [ ] SR - NN-bicubic 바꿔서 해보기

- [ ] variance map 나오는 형태 분석하기 (residual 과 차이점)

- [ ] 기존의 uncertainty 맵은 거리가 멀거나 가려져 있을 떄지만 이거는 텍스쳐 엣지 영역에서 많이 나타남

  - [ ] 도메인의 목적에 따라서 불확실성의 특성이 바뀜

  

### Method 

- [ ] sampling을 최소화 할 수 있는 방법 - MCBIN
- [ ] samping 했을 때 어떤식으로 달라지는지 확인
- [ ] Instance 와 batch norm을 섞으면 sampling 하기 쉽지 않을까
  - [ ] random하게 mask 조절. 기존은 batch 저장해서 써는데 이거는 그렇게 안해도 됨
  - [ ] BIN코드 이용해서 하기. 학습때 랜덤요소. 테스트때 비중에 따른 랜덤 샘플







- resize를 다르게 했을 때 var 커지는지 확인 (bi학습,BIi테스트vsNN테스트)
  
  - 이거는 다른 uncertainty로 측정하면 더 클 듯
  
- classification 문제와 결합햇을 때 괘안아 지는지 (machin vision 관점에서)

  ---

  질문 : Loss 왜 그렇게 나오는가-VAE 와 다른점 설명하기

  variance를 log로 취하는 이유

---

- 190508


- - EDSR train baseline (x2, x4)

  - EDSR train baseline with uncertainty (x2)

    - compare mean image
  - compare random sample image
  
  
---

- 1905015


  - [x] SR with uncertain paper

  - [x] Vggloss 일때 uncertainty 맵 확이

  - [ ] ~~얼굴데이터로 sr했을때의 uncertainty 측정~~
    ~~얼굴인식 네트워크에 confidence map 같이 적용했을때 성능 오르는지 확인~~

  - [x] loss weight 다르게 해서 학습 (loss1 + 0.01 * loss2)


    - [x] 별 차이 없음

  - [ ] 모델 1, 모델2 학습 중


    - [ ] 각각의 결과를 비교해서 달라지는 영역과 uncertainty 비교
    - [ ] (노란색 머리털 부분이 달라짐, 생각보다 털(texture부분은 변경 없음))


​    

------

- 190516
  - [ ] 학습된 모델로 BSD/Urban 돌렸을 때 양상 확인
    - [ ] line, Edge 맵에서 잘 되나?
  - [ ] Urban으로만 학습 후 DIV2K 학습
    - [ ] 다른 도메인에서 학습 했을 때 uncertatinty 가 어떻게 나타나는지 확인
  - [ ] Uncertainty 측정 방법 implementation 하기
  - [ ] Confidence맵을 어떻게 활용할 것인가
    - [ ] 어떤의미인가, edge 위주로 나오는데 이게 uncertainty가 맞나
  - [ ] 다른 uncertainty 확인
    - [ ] 다른 것은 texture 부분이 나오는지
    - [ ] 그렇다면 합쳤을 때 결과 기대


