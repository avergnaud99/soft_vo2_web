from sqlalchemy import create_engine, inspect

# Database connection information
DATABASE_URL = "sqlite:///../VO2.db"

# Create SQLAlchemy engine and inspector
engine = create_engine(DATABASE_URL)
inspector = inspect(engine)

# Generate README content
readme_content = "# Database Schema\n\n"
readme_content += "## Overview\n\n"
readme_content += "This README provides an overview of the database schema used in the project. It describes the tables, columns, and relationships.\n\n"
readme_content += "## Tables\n\n"

# Iterate over tables
for table_name in inspector.get_table_names():
    readme_content += f"### {table_name}\n\n"
    readme_content += f"- Description: TODO\n"
    readme_content += "- Columns:\n"
    
    # Get column information
    columns = inspector.get_columns(table_name)
    for column in columns:
        readme_content += f"  - `{column['name']}`: {column['type']}\n"
    
    # Get foreign keys (relationships)
    foreign_keys = inspector.get_foreign_keys(table_name)
    if foreign_keys:
        readme_content += "- Relationships:\n"
        for foreign_key in foreign_keys:
            readme_content += f"  - `{foreign_key['constrained_columns']}` references `{foreign_key['referred_table']} ({foreign_key['referred_columns']})`\n"
    
    readme_content += "\n"

# Write README content to file
with open("README.md", "w") as readme_file:
    readme_file.write(readme_content)

print("README.md successfully generated.")
