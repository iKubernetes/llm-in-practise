# 黑神话悟空评论数据集

黑神话悟空上线第一天steam评论数据集，数据集包括用户、用户链接、评价内容，是否推荐等，可用于作自然语言处理评论，数据分析用户对大圣的喜爱。

### 数据集简介

此数据集涵盖了用户信息、用户间的链接关系、对“大圣”的评价内容以及是否推荐“大圣”的标记，为进行自然语言处理和数据分析提供了丰富的素材。通过对评价内容的文本分析，可以深入挖掘用户对“大圣”的喜爱程度及其背后的原因，比如角色性格、故事情节、视觉效果等方面的吸引力。同时，结合用户间的链接关系，可以进一步分析不同用户群体对“大圣”的偏好差异，以及这种偏好如何在社交网络中传播。

user_id：用户id
user_link：用户链接
user_name：用户名
user_nationality：游戏账户国籍
user_level：用户等级
user_badge：用户徽章
user_games：用户游戏
user_achievement：游戏成就
comment_date：评论日期
recommend：是否推荐 1-推荐、0-不推荐
score：分数
comment：评论



### 部分用户信息数据集

黑神话悟空评论数据集的补充，用于协助评论权重等方面的分析。

 列信息：用户链接，用户名，用户国籍（仅用户设定的账号国籍），用户等级，用户持有徽章数，用户持有游戏数，用户持有成就数 。

使用方法：与评论数据集merge on用户链接，即内容为用户 steam 个人 profile 页的 url 其中【黑神话悟空评论数据集】。
