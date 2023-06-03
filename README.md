* 본 챌린지에서 인식률(%)의 높고 낮음은 주 평가 대상이 아닙니다. (단, 인식률이 높은 팀에게 수업 평가 외 별도의 reward는 있을 수 있습니다.)

* 결과 제출은 여러 번 가능합니다. 더 좋은 결과가 나올 때마다 제출하는 것을 추천합니다. (최다 제출 팀에게 수업 평가 외 별도의 reward가 있을 수 있습니다.)


### 데이터

* 데이터 다운로드

 * 고해상도(1920x1080, 고용량, 12GB)
  * https://drive.google.com/file/d/1E85kfIxzjPgH9-noNsbmWV_l9Qg_9Lsb/view?usp=share_link 

 * 저해상도(320x174, 저용량, 86MB)
  * https://drive.google.com/file/d/1mgJ1NyHsUB4SiDulXa3JGj5HTUkIg1Dy/view?usp=share_link
  * 편의를 위해 제공한 것입니다. 저해상도 데이터를 반드시 사용해야 할 필요는 없습니다.

* 데이터 설명
 * 데이터는 43개의 동일 종류 식물의 뿌리를 약 20분 간격으로 수 일(3일 이상)간 촬영한 이미지 들입니다.
  * 단, 촬영 일 수는 식물마다 다를 수 있습니다. 
 * 데이터는 크게 [train]과 [test] 두 폴더로 나뉩니다.
 * [train]과 [test] 하위에는 식물별로 폴더가 구분되어 있습니다.
  * 예: [root1_220823] 폴더에는 한 식물 만을 찍은 사진 들이 들어있습니다.
  * 폴더명은 크게 상관할 필요 없습니다.

 * 각 폴더에는 약 20분 간격으로 찍은 사진들이 포함되어 있습니다.
  * 파일명 양식은 다음과 같습니다. 
   * "rootx_연월일시분초.jpg"
  * 연, 월, 일, 시, 분, 초는 각각 2자리 값을 가집니다.
   * 예: "root1_220824132614.jpg"는 2022년 08월 24일 13시 26분 14초에 찍힌 사진을 의미합니다.
  * root1 또는 root2 등은 큰 의미가 없는 파일명입니다.

사진의 원 해상도는 다음과 같습니다.

1920x1080

해상도가 높으므로(용량이 크므로) 별도의 툴을 이용하여 해상도를 자유롭게 변경하여 학습에 사용하셔도 됩니다.

몇몇 폴더에는 readme.txt 파일이 들어있고, 잘 성장하지 못한 식물의 특징이 간략하게 적혀있습니다.

readme.txt 파일이 없는 경우는 정상 식물로 간주하면 됩니다.

비정상 식물의 데이터를 학습에 쓸 지 안쓸지는 자율입니다.

몇몇 폴더에는 손상된 이미지나 잘못된 이미지가 있을 수 있습니다.

손상된 이미지를 수동으로 또는 자동으로(코드로) 걸러낼 수 있는 작업이 필요합니다.

모든 식물은 동일한 시기부터 사진을 촬영하기 시작한 것이라고 생각하시면 됩니다.


목표

식물이 사진을 촬영하기 시작한 후로 12시간 단위로 얼마의 시간이 흘렀는지 분류하는 인공지능 모델 설계 및 구현

입력: 사진 이미지 1장

출력: 다음 중 하나의 카테고리

0~12시간

12~24시간

24시간~36시간

36시간~48시간

48시간~60시간

60시간~72시간

그 이상

train 데이터로 학습을 진행하고 test 데이터로 성능 검증

test 데이터의 인식률(%)을 최대가 되도록

참고

인식률이 높을 수록 좋겠지만, 데이터 특성상 인식률에는 한계가 있을 수 있습니다.

본 챌린지의 목적은 단순 인식률을 높이는 것보다는 현재 데이터로 어느 수준까지 가능한지를 확인해보고, 인식률을 높이기 위해 어떠한 것들이 개선될 수 있는지를 고민해보고, 날 것의 데이터(raw data)를 실제 인공지능 학습을 위해 어떻게 전처리(preprocessing)해야 하는지 경험(사실 이게 제일 중요)하는 것에 주 목적이 있습니다.


방법

뉴럴 네트워크, 머신 러닝, 컴퓨터 비전 등의 다양한 방식을 복합적으로 시도해볼 수 있습니다.

수업 시간에 배운 내용 이외의 것도 사용할 수 있으면 사용하시기 바랍니다.


결과 제출

아래 구글 양식을 통해 결과 제출

https://forms.gle/ochWiLVbLoEMv8yy6 

※ 여러 번 제출 가능 ※

제출할 내용

train 데이터의 인식률

test 데이터의 인식률

인식률 히스토리 그래프

GitHub 또는 Google Drive 를 이용하여 코드도 함께 제출

보고서에서는 가장 최종 코드만 첨부

추후 발표자료나 보고서는 별도 제출

발표자료와 보고서에는 수업시간에 가이드 해준 내용 이외에 다음의 내용도 넣어주세요.

현재 데이터의 한계점을 명확히 확인하고 기술합니다.

데이터 전처리 과정에서 어떠한 작업들을 했는지 자세히 적고, 전처리 과정에 코드를 썼다면 보고서에는 코드도 포함시킵니다.

시간 분류 이외에 본 이미지 데이터로 해볼 수 있는 다른 아이디어들도 제시해봅니다.
