---
layout: post
title: Sales Target & Performance Management
type: java
comments: true
categories: project
tags: [JAVA]
img: sales.jpg
---

This project aims at developing a highly cost-effective generic product which would cater to the needs of the banking sector. There is a need to have a less cost and unique application to set sales target and track performance across industry groups. Sales Target & Performance Management software will enable the organization’s senior management to set sales target at the highest level & cascade it down to the zones, clusters & branches. The actual performance will be entered by sales agents on the ground. Performance review reports/dashboards can be rolled up at various levels. It can easily be customized as per requirements and available products to suit the needs of different sector.

### SCOPE:
STPM is a web based system which was deliberated to develop as a solution for transparent and effective management of sales performance. Therefore this system acts as a supporter for executives who are re-assigning sales target based on sales progress. Under this sub topic we describe what features are in the scope of the system to be developed.

### In scope

- Transparent & Effective management of sales performance.
- Increase in revenue & productivity with stronger sales performance tracking & control.
- Bottom-up Sales projections/forecasting against products – based on product-wise performance on targets.
- Provides automated means to bank executive, top-down from top most national level to branch to set sales target.
- Review performance against the sales targets at any point in time.
- Re-Plan/Re-Assign sales target based on sales progress.
- Report sales performance to the next higher level – faster & with efficiency.
- Intangible - Increase in product revenue & productivity.

### FUNCTIONS:

- Deals with the basic problem of managing enormous data.
- It caters to the organization of data in a well structured form facilitating its maintenance and expansion.
- It allows the user to set targets about different products in a quicker and systematic way.
- Organizing and processing of data.
- Analysis, interpretation and use of assets in efficient manner.
- Help user to keep a track of all achieved target in a much quicker and efficient way.
- Provides the administrator the facility to add, modify and view different information.
- Set sales target to the products.
- Set sales target to the banking areas - zones, clusters/territory, branches etc.
- Tie sales targets to the agents.
- Agents view to update on planned & actual against set targets.
- Roll-up of performance reports at various levels with all possible data filtration requirements.

### PROPOSED SYSTEM:
Proposed system is an automated Sales Management System. Functionalities describe which services that we intended provide for the administrators and details about the functions which perform by our system STPM. Our proposed system has the following advantages.

### Provide facility for Creating an account
The system will provide facilities for the end users to create their own personal account to access the sales target and performance management. So then they can use the system for managing their business plan. Here one user can access from only one account and the user name and password should be entered correctly. So the system can provide the security of the personal data. And also when creating an account, system requests for the email address of the end user, to ensure the security of the personal data.

### Provide facility for Entering business details
The system will provide facilities for entering different details to the administrator who have already created an account in the system. Through in our system STPM, we intended to provide some categories for administrator such as,

- Create Organization levels
- Create Organization Role.
- Add Business Area.
- Add Product Category.
- Add Product.
- Add User Profile.
- Edit User Profile.
- Role to Area Mapping.
- Role to User Mapping.
- Grant Permissions.

### Set/Change Targets
- Checks the geographical level of employee from database and based on his level business areas are shown.
- Targets can be set for all the below levels.
- Targets can be set as per category and product wise as well.
- Targets can be viewed and edited.
- User can view the targets set and achieved from the previous time in the edit screen.

### Track Performance
- Agents can see the target assigned to them.
- Agents feed the actual data.
- The actual data is validated by the upper level user.

### Generate reports
System will provide various kinds of reports to the end user such as Graphical reports, textual reports and charts. To generate some reports system uses the data which are entered by the end user to the system, and sometimes system processes some data by calculations as well and uses that information for the purpose generating reports.

### Provide alert notifications
System shall provide email notifications and windows notifications for the target submitted and achieved which happen in recent future. The user can customize this option in such a way that he only gets notification as he wishes, because the user can decide which assets he needs notifications on, and which notifications or notification combination (such as windows notification and/or email notification) he needs to the particular asset. All the manual difficulties in managing the assets procedure have been rectified by implementing computerization.

### DESIGN DESCRIPTION:
> *General Flowchart of STPM*

![flowchart](https://github.com/h-shekhar/Sales-Target-Performance-Management/blob/master/Images/design.png)

> *Process Flow for STPM*

![image1](https://github.com/h-shekhar/Sales-Target-Performance-Management/blob/master/Images/Capture.JPG)

### DATABASE TABLES:

**TABLE 1- AGENT TARGET LIST**  
At_id, Target_id, Month_id, Product_id, Amt_qty, Emp_id, Approval_status

**TABLE 2- BUSINESS AREA:**  
Area_id, Area_name, Parent_area_id

**TABLE 3- EMPLOYEE:**  
Emp_id, Emp_name, Login_id, Password, Role_id, Area_id, Manager_id, Locale_status, Account_status

**TABLE 4- FINANCIAL YEAR:**  
Id, Year_from, Year_to

**TABLE 5- MAIL:**  
Mail_id, To, From, Subject, Message, Date, Status

**TABLE 6- MONTH LIST:**  
Month_id, Month_name

**TABLE 7- MONTHLY_TARGETS:**  
Mt_id, M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M12, Target_id

**TABLE 8- PRODUCTS:**  
Product_id, Product_name

**TABLE 9- QUATERLY TARGETS:**  
Qt_id, Q1, Q2, Q3, Q4, Target_id

**TABLE 10- ROLE:**  
Role_id, Role_name

**TABLE 11- TARGETS:**  
Target_id, Givenby_area_id, Year, Target, Product_id, Givento_area_id, Givenby_emp_id, Givento_emp_id

### SCREENSHOTS:
> *Fig 1: HomePage*

![a](https://github.com/h-shekhar/Sales-Target-Performance-Management/blob/master/Images/a.jpg)

> *Fig 2: Dashboard*

![e](https://github.com/h-shekhar/Sales-Target-Performance-Management/blob/master/Images/e.jpg)

> *Fig 3: New Registration*

![b](https://github.com/h-shekhar/Sales-Target-Performance-Management/blob/master/Images/b.jpg)

> *Fig 4: Set Targets*

![c](https://github.com/h-shekhar/Sales-Target-Performance-Management/blob/master/Images/c.jpg)

> *Fig 5: Approve Target List*

![d](https://github.com/h-shekhar/Sales-Target-Performance-Management/blob/master/Images/d.jpg)

> *Fig 6: View Achievements*

![f](https://github.com/h-shekhar/Sales-Target-Performance-Management/blob/master/Images/f.jpg)

> *Fig 7: Admin Panel (User List)*

![h](https://github.com/h-shekhar/Sales-Target-Performance-Management/blob/master/Images/h.jpg)

> *Fig 8: Mailbox*

![g](https://github.com/h-shekhar/Sales-Target-Performance-Management/blob/master/Images/g.jpg)

  
*Complete code can be found* [here](https://github.com/h-shekhar/Sales-Target-Performance-Management)

  
> _In case if you found something useful to add to this article or you found a bug in the code or would like to improve some points mentioned, feel free to write it down in the comments. Hope you found something useful here._


