#!/usr/bin/env python3
"""
Grading runner script for data science problem testing.

This script:
1. Creates a copy of the workspace in /tmp
2. Copies the golden script into the grading directory
3. Runs the golden script as ubuntu
4. Compares output files with expected values (string match post-trim)
"""

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class GradingRunner:
    """Handles the grading workflow for data science problems."""

    def __init__(
        self,
        template: str,
        golden_script: str,
        required_outputs: dict[str, str],
    ):
        """
        Initialize the grading runner.

        Args:
            template: The template name (subfolder in problem_templates)
            golden_script: The name of the golden script file (in /problems/)
            required_outputs: Dict mapping output file paths (relative to workspace) 
                              to expected values (string match post-trim)
        """
        self.template = template
        self.golden_script = golden_script
        self.required_outputs = required_outputs
        self.workspace_path = "/home/ubuntu/workspace"
        self.problems_dir = "/problems"
        self.grade_working_dir = "/tmp/grading_workspace_" + str(uuid.uuid4())

    def _format_junit_xml(
        self, 
        test_name: str, 
        failure_message: str | None = None, 
        stdout: str = "", 
        stderr: str = ""
    ) -> str:
        """Format a JUnit XML result."""
        failures = "1" if failure_message else "0"
        failure_xml = ""
        if failure_message:
            # Escape XML special characters
            escaped_message = (
                failure_message
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )
            failure_xml = f"<failure type='TestFailure'>\n{escaped_message}\n</failure>"
        
        escaped_stdout = stdout.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        escaped_stderr = stderr.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="{test_name}" tests="1" failures="{failures}" errors="0" skipped="0">
    <testcase classname="{test_name}" name="test{test_name}" time="0.0">
      {failure_xml}
      <system-out>\n{escaped_stdout}\n</system-out>
      <system-err>\n{escaped_stderr}\n</system-err>
    </testcase>
  </testsuite>
</testsuites>"""

    def _copy_workspace(self) -> bool:
        """Copy workspace to grading directory."""
        try:
            logger.info(f"Copying workspace from {self.workspace_path} to {self.grade_working_dir}")
            shutil.copytree(self.workspace_path, self.grade_working_dir)
            
            # Ensure ubuntu owns the grading directory
            subprocess.run(
                ["chown", "-R", "ubuntu:ubuntu", self.grade_working_dir],
                check=True
            )
            logger.info(f"Copied workspace to {self.grade_working_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy workspace: {e}")
            return False

    def _copy_golden_script(self) -> bool:
        """Copy golden script into grading directory."""
        try:
            src = Path(self.problems_dir) / self.golden_script
            dst = Path(self.grade_working_dir) / self.golden_script
            
            logger.info(f"Copying golden script from {src} to {dst}")
            shutil.copy2(src, dst)
            
            # Ensure ubuntu owns the script
            subprocess.run(["chown", "ubuntu:ubuntu", str(dst)], check=True)
            logger.info(f"Copied golden script to {dst}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy golden script: {e}")
            return False

    def _run_golden_script(self) -> tuple[bool, str, str]:
        """
        Run the golden script as ubuntu user with cwd as the grading directory.
        
        Returns:
            Tuple of (success, stdout, stderr)
        """
        script_path = Path(self.grade_working_dir) / self.golden_script
        
        logger.info(f"Running golden script: {script_path}")
        
        result = subprocess.run(
            ["sudo", "-u", "ubuntu", "python3", str(script_path)],
            cwd=self.grade_working_dir,
            capture_output=True,
            text=True,
            env=dict(os.environ, HOME="/home/ubuntu"),
        )
        
        logger.info(f"Golden script completed with code: {result.returncode}")
        logger.info(f"Golden script stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"Golden script stderr: {result.stderr}")
        
        return result.returncode == 0, result.stdout, result.stderr

    def _check_outputs(self) -> tuple[bool, dict[str, dict]]:
        """
        Check each required output file against expected value.
        
        Returns:
            Tuple of (all_passed, results_dict)
            results_dict maps filename to {passed, expected, actual, error}
        """
        results = {}
        all_passed = True
        
        for output_file, expected_value in self.required_outputs.items():
            file_path = Path(self.grade_working_dir) / output_file
            result = {
                "passed": False,
                "expected": expected_value.strip(),
                "actual": None,
                "error": None,
            }
            
            try:
                if not file_path.exists():
                    result["error"] = f"File not found: {output_file}"
                    all_passed = False
                else:
                    actual_content = file_path.read_text()
                    result["actual"] = actual_content.strip()
                    
                    # String match post-trim
                    if result["actual"] == result["expected"]:
                        result["passed"] = True
                        logger.info(f"Output check PASSED for {output_file}")
                    else:
                        result["error"] = "Content mismatch"
                        all_passed = False
                        logger.info(
                            f"Output check FAILED for {output_file}: "
                            f"expected '{result['expected']}', got '{result['actual']}'"
                        )
            except Exception as e:
                result["error"] = str(e)
                all_passed = False
                logger.error(f"Error checking output {output_file}: {e}")
            
            results[output_file] = result
        
        return all_passed, results

    def run_grading(self) -> tuple[bool, dict]:
        """
        Run the complete grading workflow.
        
        Returns:
            Tuple of (success, result_dict)
            result_dict contains 'junit' XML and other metadata
        """
        logger.info("Starting grading workflow")

        # Step 1: Copy workspace to grading directory
        if not self._copy_workspace():
            xml = self._format_junit_xml(
                "WorkspaceCopy",
                "Failed to copy workspace to grading directory"
            )
            return False, {"junit": xml}

        # Step 2: Copy golden script
        if not self._copy_golden_script():
            xml = self._format_junit_xml(
                "GoldenScriptCopy",
                f"Failed to copy golden script: {self.golden_script}"
            )
            return False, {"junit": xml}

        # Step 3: Run golden script
        script_success, stdout, stderr = self._run_golden_script()
        if not script_success:
            xml = self._format_junit_xml(
                "GoldenScriptExecution",
                "Golden script failed to execute",
                stdout,
                stderr
            )
            return False, {"junit": xml, "stdout": stdout, "stderr": stderr}

        # Step 4: Check outputs
        all_passed, output_results = self._check_outputs()
        
        # Build detailed failure message if any checks failed
        if not all_passed:
            failure_details = []
            for filename, result in output_results.items():
                if not result["passed"]:
                    if result["error"]:
                        failure_details.append(f"{filename}: {result['error']}")
                    else:
                        failure_details.append(
                            f"{filename}: expected '{result['expected']}', "
                            f"got '{result['actual']}'"
                        )
            
            failure_message = "Output validation failed:\n" + "\n".join(failure_details)
            xml = self._format_junit_xml(
                "OutputValidation",
                failure_message,
                stdout,
                stderr
            )
        else:
            xml = self._format_junit_xml("OutputValidation", None, stdout, stderr)

        return all_passed, {
            "junit": xml,
            "stdout": stdout,
            "stderr": stderr,
            "output_results": output_results,
        }

    def validate_golden(self) -> tuple[bool, dict]:
        """
        Validate that the golden script produces the expected outputs.
        
        This runs on a fresh copy of the template to verify the golden script works.
        """
        logger.info("Starting golden script validation")
        
        # For validation, we use the template directly (simulating fresh setup)
        templates_dir = "/problem_templates"
        template_path = Path(templates_dir) / self.template
        
        if not template_path.exists():
            xml = self._format_junit_xml(
                "TemplateExists",
                f"Template directory not found: {template_path}"
            )
            return False, {"junit": xml}
        
        # Copy template to grading directory
        try:
            logger.info(f"Copying template from {template_path} to {self.grade_working_dir}")
            shutil.copytree(template_path, self.grade_working_dir)
            subprocess.run(
                ["chown", "-R", "ubuntu:ubuntu", self.grade_working_dir],
                check=True
            )
        except Exception as e:
            xml = self._format_junit_xml(
                "TemplateCopy",
                f"Failed to copy template: {e}"
            )
            return False, {"junit": xml}
        
        # Copy golden script
        if not self._copy_golden_script():
            xml = self._format_junit_xml(
                "GoldenScriptCopy",
                f"Failed to copy golden script: {self.golden_script}"
            )
            return False, {"junit": xml}

        # Run golden script
        script_success, stdout, stderr = self._run_golden_script()
        if not script_success:
            xml = self._format_junit_xml(
                "GoldenScriptExecution",
                "Golden script failed to execute",
                stdout,
                stderr
            )
            return False, {"junit": xml, "stdout": stdout, "stderr": stderr}

        # Check outputs
        all_passed, output_results = self._check_outputs()
        
        if not all_passed:
            failure_details = []
            for filename, result in output_results.items():
                if not result["passed"]:
                    if result["error"]:
                        failure_details.append(f"{filename}: {result['error']}")
                    else:
                        failure_details.append(
                            f"{filename}: expected '{result['expected']}', "
                            f"got '{result['actual']}'"
                        )
            
            failure_message = "Golden script validation failed:\n" + "\n".join(failure_details)
            xml = self._format_junit_xml(
                "GoldenValidation",
                failure_message,
                stdout,
                stderr
            )
        else:
            # All passed
            xml = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="GoldenValidation" tests="3" failures="0" errors="0" skipped="0">
    <testcase classname="GoldenValidation" name="testTemplateExists" time="0.0"/>
    <testcase classname="GoldenValidation" name="testGoldenScriptRuns" time="0.0"/>
    <testcase classname="GoldenValidation" name="testOutputsMatch" time="0.0"/>
  </testsuite>
</testsuites>"""
        
        logger.info(f"Golden validation {'passed' if all_passed else 'failed'}")
        return all_passed, {
            "junit": xml,
            "stdout": stdout,
            "stderr": stderr,
            "output_results": output_results,
        }
