def test_ai_insights(session_id):
    """Test AI insights endpoint."""
    print("\n=== Testing AI Insights ===")

    if not session_id:
        print("❌ No session ID provided")
        return False

    try:
        # Test dormant account insights
        print("Testing dormant account summary insights...")
        response = requests.post(
            f"{BASE_URL}/analyze/ai-insights",
            data={
                "session_id": session_id,
                "analysis_type": "dormant",
                "insight_type": "summary"
            },
            auth=(API_USERNAME, API_PASSWORD)
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Dormant account summary insights generated successfully!")
            insight_preview = result.get('insight', '')[:150]
            print(f"Insight preview: {insight_preview}...")

            # Now test compliance insights
            print("\nTesting compliance summary insights...")
            response = requests.post(
                f"{BASE_URL}/analyze/ai-insights",
                data={
                    "session_id": session_id,
                    "analysis_type": "compliance",
                    "insight_type": "summary"
                },
                auth=(API_USERNAME, API_PASSWORD)
            )

            if response.status_code == 200:
                result = response.json()
                print(f"✅ Compliance summary insights generated successfully!")
                insight_preview = result.get('insight', '')[:150]
                print(f"Insight preview: {insight_preview}...")
                return True
            else:
                print(f"❌ Failed to generate compliance insights")
                print(f"Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
        else:
            print(f"❌ Failed to generate dormant account insights")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error generating AI insights: {str(e)}")
        traceback.print_exc()
        return False


def test_save_to_database(session_id):
    """Test saving data to database endpoint."""
    print("\n=== Testing Save to Database ===")

    if not session_id:
        print("❌ No session ID provided")
        return False

    try:
        response = requests.post(
            f"{BASE_URL}/save-to-database",
            data={
                "session_id": session_id,
                "table_name": "accounts_data"
            },
            auth=(API_USERNAME, API_PASSWORD)
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Data saved to database successfully!")
            print(f"Message: {result.get('message')}")
            return True
        else:
            print(f"❌ Failed to save data to database")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error saving data to database: {str(e)}")
        traceback.print_exc()
        return False


def test_health_check():
    """Test the health check endpoint."""
    print("\n=== Testing Health Check ===")

    try:
        response = requests.get(f"{BASE_URL}/health")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ API health check successful!")
            print(f"Status: {result.get('status')}")
            print(f"Database: {result.get('database')}")
            print(f"AI Model: {result.get('ai_model')}")
            return True
        else:
            print(f"❌ Failed to get health status")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error checking API health: {str(e)}")
        traceback.print_exc()
        return False


def test_get_data_sample(session_id):
    """Test getting data sample endpoint."""
    print("\n=== Testing Get Data Sample ===")

    if not session_id:
        print("❌ No session ID provided")
        return False

    try:
        response = requests.get(
            f"{BASE_URL}/get-data-sample/{session_id}?rows=10",
            auth=(API_USERNAME, API_PASSWORD)
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Data sample retrieved successfully!")
            print(f"Total rows in dataset: {result.get('total_rows')}")
            print(f"Columns: {', '.join(result.get('columns', []))}")
            print(f"Sample size: {len(result.get('sample', []))}")
            return True
        else:
            print(f"❌ Failed to get data sample")
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error getting data sample: {str(e)}")
        traceback.print_exc()
        return False


def run_all_tests(file_path=None):
    """Run all API tests."""
    print("\n=== Running All Banking Compliance API Tests ===")

    # Start with health check
    api_running = test_root_endpoint()
    if not api_running:
        print("❌ API is not running. Please start the API server first.")
        return

    # Check API health
    test_health_check()

    # Test database connection and query
    db_connection_id = test_database_connection()
    if db_connection_id:
        test_database_query()

    # Test file upload if file path provided
    session_id = None
    if file_path:
        session_id = test_upload_file(file_path)
    else:
        print("No file path provided, skipping file upload test.")
        return

    # Skip remaining tests if file upload failed
    if not session_id:
        print("❌ File upload failed, skipping remaining tests.")
        return

    # Test getting data sample
    test_get_data_sample(session_id)

    # Test dormant and compliance analysis
    test_dormant_analysis(session_id)
    test_compliance_analysis(session_id)

    # Test account flagging
    test_flag_accounts(session_id)

    # Test AI insights (if AI model available)
    test_ai_insights(session_id)

    # Test saving to database
    test_save_to_database(session_id)

    print("\n=== All Tests Completed ===")


if __name__ == "__main__":
    # Check if file path is provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        run_all_tests(file_path)
    else:
        print("Please provide a file path to test.")
        print("Usage: python test_api.py path/to/your/data.csv")
