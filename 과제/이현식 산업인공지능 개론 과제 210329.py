#!pip install durable_rules
from durable.lang import *

from durable.lang import _main_host

#재실행 하니까 자꾸 죽길래... 집어 넣음
if _main_host is not None:
  _main_host._ruleset_directory.clear()
#업무 절차 비젼 제어 cs 영업 구분
with ruleset('3사업부 업무절차'):
  @when_all(c.first<<(m.object == '검사 SW 문제') & (m.predicate == '접수됐다'),(m.predicate == '1년이하이다') & (m.object == '반입일이') & (m.subject == c.first.subject))
  def 검사SW문제1년이하(c):
    c.assert_fact({'subject': c.first.subject, 'object': ' 비젼 Team', 'predicate': '업무 전달'})

  @when_all(c.first<<(m.object == '검사 SW 문제') & (m.predicate == '접수됐다'),(m.predicate == '1년이상이다') & (m.object == '반입일이') & (m.subject == c.first.subject))
  def 검사SW문제1년이상(c):
    c.assert_fact({'subject': c.first.subject, 'object': ' 영업 Team', 'predicate': '업무 전달'})

  @when_all(c.first<<(m.object == '검사 SW 죽는 문제') & (m.predicate == '접수됐다'))
  def 검사SW죽는문제(c):
    c.assert_fact({'subject': c.first.subject, 'object': ' 비젼 Team 책임 이상', 'predicate': '업무 전달'})

  @when_all(c.first<<(m.object == '검사 SW 기능 추가') & (m.predicate == '접수됐다'))
  def 검사SW기능추가(c):
    c.assert_fact({'subject': c.first.subject, 'object': ' CS Team', 'predicate': '업무 전달'})

  @when_all(c.first<<(m.object == '검사 SW 알고리즘 추가') & (m.predicate == '접수됐다'))
  def 검사SW알고리즘추가(c):
    c.assert_fact({'subject': c.first.subject, 'object': ' 비젼 Team 선임 이상', 'predicate': '업무 전달'})

  @when_all(c.first<<(m.object == '부품 파손') & (m.predicate == '접수됐다'))
  def 부품파손(c):
    c.assert_fact({'subject': c.first.subject, 'object': ' CS Team 과장 이상', 'predicate': '업무 전달'})

  @when_all(c.first<<(m.object == '소모품 소모') & (m.predicate == '접수됐다'))
  def 소모품소모(c):
    c.assert_fact({'subject': c.first.subject, 'object': ' 영업 Team', 'predicate': '업무 전달'})

  @when_all(c.first<<(m.object == '제어 SW 문제') & (m.predicate == '접수됐다'),(m.predicate == '1년이하이다') & (m.object == '반입일이') & (m.subject == c.first.subject))
  def 제어SW문제1년이하(c):
    c.assert_fact({'subject': c.first.subject, 'object': ' 제어 Team', 'predicate': '업무 전달'})

  @when_all(c.first<<(m.object == '제어 SW 문제') & (m.predicate == '접수됐다'),(m.predicate == '1년이상이다') & (m.object == '반입일이') & (m.subject == c.first.subject))
  def 제어SW문제1년이상(c):
    c.assert_fact({'subject': c.first.subject, 'object': ' 영업 Team', 'predicate': '업무 전달'})

  @when_all(c.first<<(m.object == '제어 SW 죽는 문제') & (m.predicate == '접수됐다'))
  def 제어SW죽는문제(c):
    c.assert_fact({'subject': c.first.subject, 'object': ' 제어 Team 선임 이상', 'predicate': '업무 전달'})

  @when_all(c.first<<(m.object == '제어 SW 기능 추가') & (m.predicate == '접수됐다'))
  def 제어SW기능추가(c):
    c.assert_fact({'subject': c.first.subject, 'object': ' 영업 Team', 'predicate': '업무 전달'})

  @when_all(c.first<<(m.object == '이상동작') & (m.predicate == '접수됐다'))
  def 이상동작(c):
    c.assert_fact({'subject': c.first.subject, 'object': ' CS Team', 'predicate': '업무 수행'})

  @when_all((m.object == '비젼 Team') & (m.predicate == '업무 전달'))
  def 비젼팀연구원호출(c):
    c.assert_fact({'subject': c.m.subject, 'object': ' 비젼팀 연구원', 'predicate': '호출'})

  @when_all((m.object == '비젼 Team 선임이상') & (m.predicate == '업무 전달'))
  def 비젼팀선임연구원연락(c):
    c.assert_fact({'subject': c.m.subject, 'object': ' 비젼팀 선임 연구원', 'predicate': '연락'})

  @when_all((m.object == '비젼 Team 책임이상') & (m.predicate == '업무 전달'))
  def 비젼팀책임연구원연락(c):
    c.assert_fact({'subject': c.m.subject, 'object': ' 비젼팀 책임 연구원', 'predicate': '연락'})

  @when_all((m.object == '제어 Team') & (m.predicate == '업무 전달'))
  def 제어팀연구원호출(c):
    c.assert_fact({'subject': c.m.subject, 'object': ' 비젼팀 연구원', 'predicate': '호출'})

  @when_all((m.object == '제어 Team 선임이상') & (m.predicate == '업무 전달'))
  def 제어팀선임연구원연락(c):
    c.assert_fact({'subject': c.m.subject, 'object': ' 비젼팀 선임 연구원', 'predicate': '연락'})

  @when_all((m.object == '제어 Team 책임이상') & (m.predicate == '업무 전달'))
  def 제어팀책임연구원연락(c):
    c.assert_fact({'subject': c.m.subject, 'object': ' 비젼팀 책임 연구원', 'predicate': '연락'})
     
  @when_all((m.object == '영업 Team') & (m.predicate == '업무 전달'))
  def 영업팀업무인계(c):
    c.assert_fact({'subject': c.m.subject, 'object': ' 영업팀 직원', 'predicate': '업무 인계'})

  @when_all((m.object == 'CS Team과장이상') & (m.predicate == '업무 전달'))
  def CS팀과장연락(c):
    c.assert_fact({'subject': c.m.subject, 'object': ' 비젼팀 연구원', 'predicate': '연락'})

  @when_all((m.object == 'CS Team') & (m.predicate == '업무 수행'))
  def CS팀업무수행(c):
    c.assert_fact({'subject': c.m.subject, 'object': ' CS대리이하', 'predicate': '업무수행'})
# 단순 프린트문
  @when_all(+m.subject)
  def output(c):
    print('Fact: {0} {1} {2}'.format(c.m.subject, c.m.object, c.m.predicate))

#특정 경우만 워런티를 물어본다
def QuestionWarranty(item):
  warranty =  input("장비 출하 후 1년 이상입니까? (0 : NO, 1 : YES)")
  if int(warranty)==0:
    assert_fact('3사업부 업무절차', { 'subject': item, 'object': '반입일이', 'predicate': '1년이하이다' })
  elif int(warranty)==1:
    assert_fact('3사업부 업무절차', { 'subject': item, 'object': '반입일이', 'predicate': '1년이상이다' })
  else:
    print('잘못 입력하였습니다. 다시 입력하세요. 워런티 ')
    QuestionWarranty()
#프로젝트를 적고 해당 문제들을 선택한다
def QuestionProblem(item):
  job = input("어떤 일입니까? (0 : SW 1 : 그외) = ") 
  if int(job) == 0:
    swType = input("어떤 파트 일 입니까 (0 : 제어 1 : 비젼) = ")
    if int(swType) == 0 :
      problem = input("어떤 문제 입니까 (0 : 죽는 문제, 1 :  기능추가, 2 : 기타) = ")
      if int(problem) == 0:
        assert_fact('3사업부 업무절차', { 'subject': item, 'object': '제어 SW 죽는 문제', 'predicate': '접수됐다' })
      elif int(problem) == 1:
        assert_fact('3사업부 업무절차', { 'subject': item, 'object': '제어 SW 기능 추가', 'predicate': '접수됐다' })
      elif int(problem) == 2:
        assert_fact('3사업부 업무절차', { 'subject': item, 'object': '제어 SW 문제', 'predicate': '접수됐다' })
        QuestionWarranty(str(item))
      else:
        print('잘못 입력하였습니다. 다시 입력하세요. 제어 문제 ')
        QuestionProblem()
    elif int(swType) == 1 :
      problem = input("어떤 문제 입니까 (0 : 죽는 문제, 1 : 알고리즘 추가, 2 : 기능추가, 3: 기타) = ")
      if int(problem) == 0:
        assert_fact('3사업부 업무절차', { 'subject': item, 'object': '검사 SW 죽는 문제', 'predicate': '접수됐다' })
      elif int(problem) == 1:
        assert_fact('3사업부 업무절차', { 'subject': item, 'object': '검사 SW 알고리즘 추가', 'predicate': '접수됐다' })
      elif int(problem) == 2:
        assert_fact('3사업부 업무절차', { 'subject': item, 'object': '검사 SW 기능 추가', 'predicate': '접수됐다' })
      elif int(problem) == 3:
        assert_fact('3사업부 업무절차', { 'subject': item, 'object': '검사 SW 문제', 'predicate': '접수됐다' })
        QuestionWarranty(str(item))
      else:
        print('잘못 입력하였습니다. 다시 입력하세요. 비젼 문제')
        QuestionProblem(item)
  if int(job) == 1:
    problem = input("어떤 문제 입니까 (0 : 부품 파손 , 1 :  소모품 소모, 2 : 이상동작) = ")
    if int(problem) == 0:
      assert_fact('3사업부 업무절차', { 'subject': item, 'object': '부품 파손', 'predicate': '접수됐다' })
    elif int(problem) == 1:
      assert_fact('3사업부 업무절차', { 'subject': item, 'object': '소모품 소모', 'predicate': '접수됐다' })
    elif int(problem) == 2:
      assert_fact('3사업부 업무절차', { 'subject': item, 'object': '이상동작', 'predicate': '접수됐다' })
    else:
      print('잘못 입력하였습니다. 다시 입력하세요. 기타 문제')
      QuestionProblem(item)


  #와일문으로 계속 등록할수있다. 동일한 프로젝트는 터짐...
while 1:
  item =  input("프로젝트를 입력하세요 : (종료하려면 quit 를 입력하세요. 동일한 프로젝트를 입력하면 죽어요...)\r\n")
  if str(item) == 'quit':
    break
  
  QuestionProblem(str(item))
  #ruleset(str(item)).reset() 죽어서 시도해보려 했으나 이거 써도 죽는다...