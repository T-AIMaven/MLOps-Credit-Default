# Databricks notebook source
# MAGIC %md
# MAGIC ### Create a query that checks the accuracy being lower than 42%

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

srcs = w.data_sources.list()


alert_query = """
    SELECT
        accuracy_score * 100 as accuracy_percentage
    FROM credit.default.model_monitoring_profile_metrics"""


# Create the query
query = w.queries.create(
    query=sql.CreateQueryRequestQuery(
        display_name=f"credit-default-accuracy-alert-query-{time.time_ns()}",
        warehouse_id=srcs[0].warehouse_id,
        description="Alert on credit default model accuracy",
        query_text=alert_query,
    )
)

# Create the alert
alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(column=sql.AlertOperandColumn(name="accuracy_percentage")),
            op=sql.AlertOperator.LESS_THAN,  # Alert when accuracy drops below threshold
            threshold=sql.AlertConditionThreshold(
                value=sql.AlertOperandValue(double_value=42)  # Alert if accuracy drops below 42%
            ),
        ),
        display_name=f"credit-default-accuracy-alert-{time.time_ns()}",
        query_id=query.id,
    )
)


# COMMAND ----------

# cleanup
w.queries.delete(id=query.id)
w.alerts.delete(id=alert.id)
